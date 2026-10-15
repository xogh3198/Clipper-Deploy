import contextlib
import base64
import io
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import sys
import types
import unittest
from unittest.mock import Mock, patch

from clipper_gpu import BASE_COMMIT
from clipper_gpu import client, deploy
from clipper_gpu.config import ROOT, compose_spec, load, read_env
from clipper_gpu.worker import GPUModel, Recorder
from clipper_gpu.rpc_bridge import load_rpc


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        c = load()
        self.assertEqual(c["GPU"], "2")
        self.assertEqual(c["SLO_MS"], "200")
        self.assertEqual(c["MODEL_NAME"], "resnet-50")
        self.assertEqual(c["BATCH_SIZE"], "-1")
        self.assertEqual(len(c["MODEL_REVISION"]), 40)
        self.assertEqual(c["BUILD_LIB_BASE"], "1")
        self.assertEqual(c["PULL_LIB_BASE"], "0")
        self.assertEqual(c["LIB_BASE_IMAGE"], "taeho-clipper/lib_base:local-v1")

    def test_literal_env_no_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.env"
            path.write_text("GPU=2\nCOMMAND=$(touch should-not-exist)\n", encoding="utf-8")
            self.assertEqual(read_env(path)["COMMAND"], "$(touch should-not-exist)")
            self.assertFalse((Path(directory) / "should-not-exist").exists())

    def test_environment_override(self):
        with patch.dict(os.environ, {"SLO_MS": "500", "GPU": "3"}):
            self.assertEqual(load()["SLO_MS"], "500")
            self.assertEqual(load()["GPU"], "3")

    def test_reject_bad_settings(self):
        for values in ({"GPU": "all"}, {"BATCH_SIZE": "0"}, {"MODEL_REVISION": "main"},
                       {"SLO_MS": "nan"}, {"QUERY_PORT": "70000"}, {"APPLICATION": "../other"},
                       {"BUILD_LIB_BASE": "yes"}, {"BASE_BUILD_JOBS": "0"}):
            with self.subTest(values=values), patch.dict(os.environ, values):
                with self.assertRaises(ValueError):
                    load()

    def test_isolated_compose(self):
        spec = compose_spec(load(), "0:0", "test-run")
        self.assertEqual(set(spec["services"]), {"redis", "management", "query", "worker"})
        query = spec["services"]["query"]
        self.assertIn("--prediction_cache_size=0", query["command"])
        self.assertTrue(all(port.startswith("127.0.0.1:") for s in spec["services"].values()
                            for port in s.get("ports", [])))
        worker = spec["services"]["worker"]
        self.assertEqual(worker["environment"]["CLIPPER_IP"], "query")
        self.assertEqual(worker["environment"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(worker["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"], ["2"])
        self.assertTrue(worker["volumes"][0]["read_only"])
        self.assertNotIn("docker.sock", json.dumps(spec))
        self.assertNotIn("infaas", json.dumps(spec).lower())

    def test_rootless_uid_mapping(self):
        self.assertEqual(deploy.docker_user({"SecurityOptions": ["name=rootless"]}), "0:0")
        with patch.object(os, "getuid", return_value=1000, create=True), \
             patch.object(os, "getgid", return_value=1000, create=True):
            self.assertEqual(deploy.docker_user({"SecurityOptions": []}), "1000:1000")

    def test_original_sources_only(self):
        dockerfile = (ROOT / "deploy/docker/worker.Dockerfile").read_text()
        self.assertIn("COPY containers/python/rpc.py", dockerfile)
        self.assertIn(BASE_COMMIT, dockerfile)
        self.assertNotIn("infaas", dockerfile.lower())
        script = (ROOT / "deploy/docker/clipper-docker.sh").read_bytes()
        self.assertNotIn(b"\r", script)
        self.assertNotIn(b"DOCKER_HOST=", script)
        self.assertNotIn(b"DOCKER_CONTEXT=", script)


class BaseImageTests(unittest.TestCase):
    def test_local_dependency_build_never_pulls_missing_tag(self):
        c = load()
        # An old user override must not re-enable the obsolete pull path.
        c["PULL_LIB_BASE"] = "1"
        with patch.object(deploy, "run") as mocked, contextlib.redirect_stdout(io.StringIO()):
            registry, version = deploy.prepare_base(c)
        self.assertEqual((registry, version), ("taeho-clipper", "local-v1"))
        mocked.assert_called_once_with(
            ["docker", "build", "--build-arg", "BUILD_JOBS=2", "-t", c["LIB_BASE_IMAGE"],
             "-f", "deploy/docker/lib-base.Dockerfile", "."], cwd=c["ROOT"])

    def test_explicit_existing_image_mode(self):
        c = load()
        c.update(BUILD_LIB_BASE="0", PULL_LIB_BASE="0")
        with patch.object(deploy, "run") as mocked:
            deploy.prepare_base(c)
        mocked.assert_called_once_with(["docker", "image", "inspect", c["LIB_BASE_IMAGE"]], capture=True)

    def test_explicit_pull_mode(self):
        c = load()
        c.update(BUILD_LIB_BASE="0", PULL_LIB_BASE="1")
        with patch.object(deploy, "run") as mocked:
            deploy.prepare_base(c)
        mocked.assert_called_once_with(["docker", "pull", c["LIB_BASE_IMAGE"]])

    def test_pinned_recipe_and_small_build_context(self):
        recipe = (ROOT / "deploy/docker/lib-base.Dockerfile").read_text()
        self.assertIn("94a77ab3adee9f3ab184e23fe0c38d4cb5086635", recipe)
        self.assertIn("8af9b8c2b889d80c22d6bc26ba0df1afb79a30db", recipe)
        self.assertIn("checkout --detach FETCH_HEAD", recipe)
        self.assertIn("https://dist.schmorp.de/libev/Attic/libev-4.25.tar.gz", recipe)
        self.assertNotIn("--no-check-certificate", recipe)
        self.assertEqual((ROOT / "deploy/docker/lib-base.Dockerfile.dockerignore").read_text().splitlines(),
                         ["*", "!deploy/", "!deploy/docker/", "!deploy/docker/fix-event-headers.py",
                          "!deploy/docker/event-header-probe.cpp"])

    def test_googletest_is_pinned_after_cached_dependencies(self):
        recipe = (ROOT / "deploy/docker/lib-base.Dockerfile").read_text()
        self.assertIn("ARG GOOGLETEST_REVISION=2fe3bd994b3189899d93f1d5a881e725e046fdc2", recipe)
        self.assertIn('"${GOOGLETEST_REVISION}:refs/heads/master"', recipe)
        self.assertIn('rev-parse master)', recipe)
        self.assertIn('git config --system url."file:///opt/clipper-build-deps/googletest.git".insteadOf', recipe)
        self.assertGreater(recipe.index("ARG GOOGLETEST_REVISION"), recipe.index("&& ldconfig"))
        self.assertIn("CMakeLists.txt", deploy.CORE_PATHS)
        self.assertIn("cmake/DownloadProject.cmake", deploy.CORE_PATHS)

    def test_modified_original_cmake_is_rejected(self):
        for changed in ("CMakeLists.txt", "cmake/DownloadProject.cmake"):
            with self.subTest(changed=changed), patch.object(deploy, "run", side_effect=["", changed]):
                with self.assertRaisesRegex(RuntimeError, "Original source differs"):
                    deploy.check_original(load())

    @unittest.skipUnless(shutil.which("git"), "Git executable required for local mirror regression")
    def test_obsolete_master_download_uses_pinned_image_local_mirror(self):
        # No network: an isolated Git config simulates the image-only URL map.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = os.environ.copy()
            for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
                env.pop(key, None)
            env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=str(root / "gitconfig"),
                       GIT_CEILING_DIRECTORIES=str(root))

            def git(*args, cwd=root):
                result = subprocess.run([shutil.which("git"), *map(str, args)], cwd=cwd, env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout.strip()

            source = root / "upstream"
            git("init", source)
            (source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 2.8.8)\n")
            git("add", "CMakeLists.txt", cwd=source)
            git("-c", "user.name=Local Test", "-c", "user.email=test@local.invalid",
                "commit", "-m", "Pinned fixture", cwd=source)
            revision = git("rev-parse", "HEAD", cwd=source)
            mirror = root / "image" / "googletest.git"
            mirror.parent.mkdir()
            git("init", "--bare", mirror)
            git("fetch", "--depth", "1", source.as_uri(), revision + ":refs/heads/master", cwd=mirror)
            git("symbolic-ref", "HEAD", "refs/heads/master", cwd=mirror)
            self.assertEqual(git("rev-parse", "master", cwd=mirror), revision)
            original_url = "https://github.com/google/googletest.git"
            git("config", "--global", "url." + mirror.as_uri() + ".insteadOf", original_url)
            downloaded = root / "googletest-src"
            git("clone", "--no-checkout", original_url, downloaded)
            git("checkout", "master", cwd=downloaded)
            self.assertEqual(git("rev-parse", "HEAD", cwd=downloaded), revision)
            self.assertEqual((downloaded / "CMakeLists.txt").read_text(),
                             "cmake_minimum_required(VERSION 2.8.8)\n")


class EventHeaderCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("event_header_compatibility",
                                                     ROOT / "deploy/docker/fix-event-headers.py")
        cls.compat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.compat)

    def fixture(self, root):
        files = {
            "usr/include/event2/event.h":
                "#define EV_TIMEOUT 0x01\n#define EV_READ 0x02\n#define EV_WRITE 0x04\n#define EV_SIGNAL 0x08\n"
                "#define EV_PERSIST 0x10\n#define EV_ET 0x20\n#define EV_FINALIZE 0x40\n#define EV_CLOSED 0x80\n"
                "#define evtimer_pending(ev, tv) event_pending(ev, EV_TIMEOUT, tv)\n",
            "usr/include/event.h": '#include <event2/event.h>\n/* EV_READ */\n',
            "usr/local/include/folly/EventHandler.h":
                "enum { READ = EV_READ, WRITE = EV_WRITE, SIGNAL = EV_SIGNAL };\nint MY_EV_READ_VALUE;\n"
                "short timeout_flag = EV_TIMEOUT;\n",
            "usr/local/include/folly/Other.h": "int unrelated;\n",
            "usr/local/include/ev.h":
                "enum { EV_READ = 0x01, EV_WRITE = 0x02, EV_TIMER = 0x00000100, "
                "EV_TIMEOUT = EV_TIMER, EV_SIGNAL = 0x00000400, EV_NONE = 0x00 };\n"
                "/* enum { EV_PERSIST = 0x10 }; comments do not declare enum names */\n",
            "usr/include/hiredis/adapters/libev.h": "ev_io_start(loop, EV_READ | EV_WRITE);\n",
            "clipper/src/query_processor.cpp": "// original source remains untouched\nEV_READ;\n",
        }
        for name, contents in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        return {name: (root / name).read_bytes() for name in files}

    def test_namespacing_preserves_values_and_original_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.fixture(root)
            report = self.compat.patch_headers(root)
            self.assertEqual(len(report["changed_headers"]), 3)
            text = (root / "usr/include/event2/event.h").read_text()
            for name, value in (("EV_TIMEOUT", 1), ("EV_READ", 2), ("EV_WRITE", 4), ("EV_SIGNAL", 8)):
                self.assertEqual(self.compat.define_value(text, "CLIPPER_LIBEVENT_" + name), value)
            self.assertIn("event_pending(ev, CLIPPER_LIBEVENT_EV_TIMEOUT, tv)", text)
            self.assertEqual(self.compat.define_value(text, "EV_PERSIST"), 0x10)
            self.assertEqual(self.compat.define_value(text, "EV_ET"), 0x20)
            self.assertEqual(self.compat.define_value(text, "EV_FINALIZE"), 0x40)
            self.assertEqual(self.compat.define_value(text, "EV_CLOSED"), 0x80)
            folly = (root / "usr/local/include/folly/EventHandler.h").read_text()
            self.assertIn("READ = CLIPPER_LIBEVENT_EV_READ", folly)
            self.assertIn("timeout_flag = CLIPPER_LIBEVENT_EV_TIMEOUT", folly)
            self.assertIn("MY_EV_READ_VALUE", folly)
            for name in ("usr/local/include/ev.h", "usr/include/hiredis/adapters/libev.h",
                         "usr/local/include/folly/Other.h", "clipper/src/query_processor.cpp"):
                self.assertEqual((root / name).read_bytes(), original[name])
            stored = json.loads((root / "opt/clipper-base-sources/event-header-compat.json").read_text())
            self.assertEqual(stored["compatibility"], "libevent-prefix-v2")
            self.assertEqual(stored["libev_values"]["EV_TIMEOUT"], 256)
            self.assertEqual(stored["libev_values"]["EV_SIGNAL"], 1024)
            self.assertIn("EV_TIMEOUT", stored["checked_libev_enum_names"])
            self.assertNotIn("EV_PERSIST", stored["checked_libev_enum_names"])
            self.assertTrue(all(len(row["before_sha256"]) == 64 for row in stored["changed_headers"]))

    def test_second_run_is_idempotent_and_keeps_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.compat.patch_headers(root)
            audit = root / "opt/clipper-base-sources/event-header-compat.json"
            before = audit.read_bytes()
            report = self.compat.patch_headers(root)
            self.assertEqual(report["changed_headers"], [])
            self.assertEqual(audit.read_bytes(), before)
            self.assertNotIn("CLIPPER_LIBEVENT_CLIPPER_LIBEVENT_",
                             (root / "usr/local/include/folly/EventHandler.h").read_text())

    def test_unexpected_constants_fail_before_changes(self):
        for name, old, new in (("usr/include/event2/event.h", "0x02", "0x10"),
                               ("usr/include/event2/event.h", "EV_TIMEOUT 0x01", "EV_TIMEOUT 0x10"),
                               ("usr/local/include/ev.h", "0x01", "0x10"),
                               ("usr/local/include/ev.h", "0x00000100", "0x00000001"),
                               ("usr/local/include/ev.h", "EV_TIMEOUT = EV_TIMER", "EV_TIMEOUT = EV_READ"),
                               ("usr/local/include/ev.h", "0x00000400", "0x00000008")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                original = self.fixture(root)
                path = root / name
                path.write_text(path.read_text().replace(old, new))
                with self.assertRaisesRegex(ValueError, "Unexpected"):
                    self.compat.patch_headers(root)
                folly = "usr/local/include/folly/EventHandler.h"
                self.assertEqual((root / folly).read_bytes(), original[folly])
                self.assertFalse((root / "opt/clipper-base-sources/event-header-compat.json").exists())

    def test_all_current_name_collisions_are_covered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.fixture(root)
            names = self.compat.enum_names(original["usr/local/include/ev.h"].decode())
            macros = self.compat.macro_names(original["usr/include/event2/event.h"].decode())
            self.assertEqual(names & macros, {"EV_TIMEOUT", "EV_READ", "EV_WRITE", "EV_SIGNAL"})
            self.compat.patch_headers(root)
            self.assertEqual(names & self.compat.macro_names(
                (root / "usr/include/event2/event.h").read_text()), set())

    def test_unknown_collision_in_any_header_fails_before_writes(self):
        for name in ("usr/include/event2/event.h", "usr/local/include/folly/Other.h"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root)
                path = root / name
                path.write_text(path.read_text() + "#define EV_TIMER 0x10\n")
                before = {entry: entry.read_bytes() for entry in root.rglob("*.h")}
                with self.assertRaisesRegex(ValueError, "Uncovered.*EV_TIMER"):
                    self.compat.patch_headers(root)
                for entry, contents in before.items():
                    self.assertEqual(entry.read_bytes(), contents)
                self.assertFalse((root / "opt/clipper-base-sources/event-header-compat.json").exists())

    def test_upgrade_v1_headers_including_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            v1_tokens = re.compile(r"\b(?:EV_READ|EV_WRITE|EV_SIGNAL)\b")
            for name in ("usr/include/event2/event.h", "usr/include/event.h",
                         "usr/local/include/folly/EventHandler.h"):
                path = root / name
                path.write_text(v1_tokens.sub(lambda match: "CLIPPER_LIBEVENT_" + match.group(0),
                                             path.read_text()))
            report = self.compat.patch_headers(root)
            self.assertEqual(report["compatibility"], "libevent-prefix-v2")
            self.assertEqual(len(report["changed_headers"]), 2)
            self.assertIn("#define CLIPPER_LIBEVENT_EV_TIMEOUT 0x01",
                          (root / "usr/include/event2/event.h").read_text())
            self.assertNotIn("CLIPPER_LIBEVENT_CLIPPER_LIBEVENT_",
                             (root / "usr/local/include/folly/EventHandler.h").read_text())

    def test_build_runs_real_compiler_probes_both_orders(self):
        recipe = (ROOT / "deploy/docker/lib-base.Dockerfile").read_text()
        self.assertIn("python3 /opt/clipper-build-deps/fix-event-headers.py", recipe)
        self.assertIn("g++ -std=c++14 -fsyntax-only", recipe)
        self.assertIn("g++ -std=c++14 -DCLIPPER_EV_FIRST -fsyntax-only", recipe)
        self.assertGreater(recipe.index("COPY deploy/docker/fix-event-headers.py"),
                           recipe.index("ARG GOOGLETEST_REVISION"))
        probe = (ROOT / "deploy/docker/event-header-probe.cpp").read_text()
        self.assertIn("static_assert(EV_READ == 0x01", probe)
        self.assertIn("static_assert(EV_TIMEOUT == EV_TIMER", probe)
        self.assertIn("static_assert(EV_SIGNAL == 0x400", probe)
        self.assertIn("static_assert(CLIPPER_LIBEVENT_EV_TIMEOUT == 0x01", probe)
        self.assertIn("static_assert(CLIPPER_LIBEVENT_EV_READ == 0x02", probe)
        self.assertIn('org.clipper.gpu.event-header-compatibility="libevent-prefix-v2"', recipe)
        self.assertIn("libevent macros must not shadow libev enum names", probe)


class FrontendBuildContextTests(unittest.TestCase):
    def test_all_git_metadata_excluded_but_sources_kept(self):
        for name in ("QueryFrontendDockerfile", "ManagementFrontendDockerfile"):
            with self.subTest(name=name):
                patterns = (ROOT / "dockerfiles" / (name + ".dockerignore")).read_text().splitlines()
                self.assertIn(".git", patterns)
                self.assertIn("**/.git", patterns)
                self.assertNotIn("src", patterns)
                self.assertNotIn("src/libs", patterns)
                self.assertNotIn("*", patterns)

    @unittest.skipUnless(shutil.which("git"), "Git executable required for packaging regression")
    def test_original_style_git_apply_without_dangling_submodule_pointer(self):
        # Reproduce COPY's dangling submodule .git pointer; then simulate the
        # corrected build context without Git metadata. Sources/patches stay.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "src" / "libs"
            dependency = source / "spdlog"
            header = dependency / "include" / "spdlog" / "common.h"
            header.parent.mkdir(parents=True)
            header.write_text("old declaration\n", encoding="utf-8")
            (dependency / ".git").write_text(
                "gitdir: ../../../.git/modules/src/libs/spdlog\n", encoding="utf-8")
            patches = source / "patches"
            patches.mkdir()
            (patches / "example.patch").write_text(
                "diff --git a/include/spdlog/common.h b/include/spdlog/common.h\n"
                "--- a/include/spdlog/common.h\n"
                "+++ b/include/spdlog/common.h\n"
                "@@ -1 +1 @@\n-old declaration\n+new declaration\n", encoding="utf-8")
            command = [shutil.which("git"), "apply", "../patches/example.patch"]
            env = os.environ.copy()
            for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
                env.pop(key, None)
            env["GIT_CEILING_DIRECTORIES"] = str(root)
            broken = subprocess.run(command, cwd=dependency, env=env, capture_output=True, text=True)
            self.assertNotEqual(broken.returncode, 0)
            self.assertIn("not a git repository", broken.stderr)
            self.assertEqual(header.read_text(encoding="utf-8"), "old declaration\n")
            build = root / "build" / "src" / "libs"
            shutil.copytree(source, build, ignore=shutil.ignore_patterns(".git"))
            fixed = subprocess.run(command, cwd=build / "spdlog", env=env, capture_output=True, text=True)
            self.assertEqual(fixed.returncode, 0, fixed.stderr)
            self.assertEqual((build / "spdlog/include/spdlog/common.h").read_text(encoding="utf-8"),
                             "new declaration\n")


class ClientTests(unittest.TestCase):
    def test_default_not_inference(self):
        self.assertIsNone(client.parse_prediction({"default": True, "output": "fallback"}, "id"))

    def test_real_inference_requires_id(self):
        result = {"request_id": "id", "label": "dog"}
        self.assertEqual(client.parse_prediction({"default": False, "output": result}, "id"), result)
        self.assertEqual(client.parse_prediction({"default": False, "output": json.dumps(result)}, "id"), result)
        with self.assertRaises(ValueError):
            client.parse_prediction({"default": False, "output": result}, "different-id")
        with self.assertRaises(ValueError):
            client.parse_prediction({"output": "dog"}, "id")

    def test_same_jpeg_unique_requests(self):
        payloads = []

        def fake_http(url, payload, timeout):
            item = json.loads(payload["input"])
            payloads.append(item)
            return {"default": False, "output": {"request_id": item["request_id"], "label": "dog"}}

        with patch.object(client, "http_json", side_effect=fake_http):
            a = client.query_one(load(), b"same-jpeg")
            b = client.query_one(load(), b"same-jpeg")
        self.assertNotEqual(a["request_id"], b["request_id"])
        self.assertEqual(payloads[0]["image_b64"], payloads[1]["image_b64"])
        self.assertEqual(a["image_sha256"], b["image_sha256"])

    def test_completed_batch(self):
        event = {"request_ids": ["a", "b"], "batch_size": 2}
        self.assertEqual(client.completed_batch({"recent_batches": [event]}, "b"), event)
        self.assertIsNone(client.completed_batch({"recent_batches": [event]}, "c"))

    def test_measure_cold_default_and_warm(self):
        with tempfile.TemporaryDirectory() as directory:
            c = load()
            c["INPUT_IMAGE"] = str(Path(directory) / "test.jpg")
            c["RESULT_DIR"] = directory
            Path(c["INPUT_IMAGE"]).write_bytes(b"jpeg")
            events = []

            def fake_state(_):
                return {"run_id": "run", "model_loaded": bool(events), "forward_calls": len(events),
                        "startup_ms": 5000, "startup_stages": {"framework_import_ms": 4900, "cuda_setup_ms": 100},
                        "cold_definition": "model non-resident; framework/CUDA initialized; caches retained",
                        "warmup_queries": 10, "warmup_forward_calls": 10 if events else 0,
                        "errors": 0, "recent_batches": list(events)}

            def fake_query(_, raw, phase):
                i = str(len(events))
                event = {"request_ids": [i], "batch_size": 1, "load_ms": 700 if not events else 0,
                         "worker_total_ms": 780 if not events else 70,
                         "load_stages": {"checkpoint_read_ms": 500, "model_h2d_ms": 50, "warmup_ms": 150}
                         if not events else {}}
                default = not events
                events.append(event)
                return {"request_id": i, "phase": phase, "default": default,
                        "real_prediction": not default, "e2e_ms": 200 if default else 70}

            with patch.object(client, "state", side_effect=fake_state), \
                 patch.object(client, "query_one", side_effect=fake_query), contextlib.redirect_stdout(io.StringIO()):
                summary = client.measure(c, count=11, cold=True)
            self.assertEqual(summary["requests"], 11)
            self.assertEqual(summary["default_responses"], 1)
            self.assertEqual(summary["warm_requests"], 10)
            self.assertEqual(summary["warm_mean_ms"], 70)
            self.assertEqual(summary["warm_load_ms"], [0] * 10)
            self.assertTrue(summary["all_request_ids_executed"])
            self.assertEqual(summary["runtime_startup_ms"], 5000)
            self.assertEqual(summary["cold_client_e2e_ms"], 200)
            self.assertTrue(summary["cold_default_response"])
            self.assertEqual(summary["cold_worker_total_ms"], 780)
            self.assertEqual(summary["cold_load_ms"], 700)
            self.assertEqual(summary["cold_load_stages"]["warmup_ms"], 150)
            self.assertEqual(summary["warmup_forward_delta"], 10)
            self.assertEqual(len(list(Path(directory).glob("client-*.json"))), 1)

    def test_reject_warm_as_cold(self):
        with tempfile.TemporaryDirectory() as directory:
            c = load()
            c["INPUT_IMAGE"] = str(Path(directory) / "test.jpg")
            Path(c["INPUT_IMAGE"]).write_bytes(b"jpeg")
            with patch.object(client, "state", return_value={"model_loaded": True}):
                with self.assertRaises(RuntimeError):
                    client.measure(c, cold=True)


class RecorderTests(unittest.TestCase):
    def test_worker_preserves_native_batch_one_forward(self):
        fake_pil = types.ModuleType("PIL")
        image = Mock()
        image.convert.return_value = "decoded RGB"
        fake_pil.Image = types.SimpleNamespace(open=Mock(return_value=contextlib.nullcontext(image)))
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            rec = Recorder(directory, "run")
            model = GPUModel.__new__(GPUModel)
            model.recorder = rec
            model.ensure_loaded = Mock(return_value=0.0)
            model.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None),
                                               inference_mode=contextlib.nullcontext)
            tensor = Mock()
            model.processor = Mock(return_value={"pixel_values": tensor})
            logits = Mock()
            logits.argmax.return_value.cpu.return_value.tolist.return_value = [0, 0, 0]
            model.model = Mock(return_value=types.SimpleNamespace(logits=logits))
            model.model.config = types.SimpleNamespace(id2label={0: "dog"})
            inputs = [json.dumps({"request_id": str(i), "image_b64": base64.b64encode(b"jpeg").decode()}).encode()
                      for i in range(3)]
            with patch.dict(sys.modules, {"PIL": fake_pil}), \
                 patch.dict(os.environ, {"CLIPPER_MODEL_NAME": "resnet-50"}):
                outputs = [json.loads(output) for output in model.predict_strings(inputs)]
            self.assertEqual([output["request_id"] for output in outputs], ["0", "1", "2"])
            self.assertTrue(all(output["batch_size"] == 3 for output in outputs))
            model.model.assert_called_once()
            tensor.to.assert_called_once_with("cuda:0")
            self.assertEqual(rec.snapshot()["max_observed_batch"], 3)

    def test_native_batch_count(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            rec = Recorder(directory, "run")
            rec.append({"event": "batch_complete", "request_ids": ["a", "b"], "batch_size": 2})
            rec.append({"event": "batch_error", "batch_size": 1, "error": "example"})
            state = rec.snapshot()
            self.assertEqual(state["forward_calls"], 1)
            self.assertEqual(state["executed_inputs"], 2)
            self.assertEqual(state["max_observed_batch"], 2)
            self.assertEqual(state["errors"], 1)
            self.assertFalse(state["model_loaded"])
            lines = rec.path.read_text().splitlines()
            self.assertEqual(len(lines), 2)


class RPCBridgeTests(unittest.TestCase):
    def test_only_legacy_metrics_import_shim(self):
        spec = types.SimpleNamespace(loader=Mock())
        module = types.ModuleType("synthetic_rpc")
        with patch.dict(sys.modules, {}, clear=False), \
             patch("clipper_gpu.rpc_bridge.importlib.util.spec_from_file_location", return_value=spec), \
             patch("clipper_gpu.rpc_bridge.importlib.util.module_from_spec", return_value=module):
            sys.modules.pop("clipper_admin", None)
            sys.modules.pop("clipper_admin.metrics", None)
            self.assertIs(load_rpc("/original/rpc.py"), module)
            spec.loader.exec_module.assert_called_once_with(module)
            metrics = sys.modules["clipper_admin.metrics"]
            with self.assertRaises(RuntimeError):
                metrics.add_metric("unexpected")


class RegistrationTests(unittest.TestCase):
    def test_first_registration_payload(self):
        c = load()
        calls = []
        app_created = False

        def fake_http(url, payload):
            nonlocal app_created
            name = url.split("/")[-1]
            calls.append((name, payload))
            if name == "get_all_applications":
                return [{"name": "imagenet", "linked_models": []}] if app_created else []
            if name == "get_all_models":
                return []
            if name == "add_app":
                app_created = True
            return "ok"

        with patch.object(client, "http_json", side_effect=fake_http), contextlib.redirect_stdout(io.StringIO()):
            deploy.register(c)
        self.assertIn(("add_app", {"name": "imagenet", "input_type": "strings",
                                  "default_output": "CLIPPER_SLO_FALLBACK", "latency_slo_micros": 200000}), calls)
        self.assertEqual(next(p for n, p in calls if n == "add_model")["batch_size"], -1)
        self.assertIn(("add_model_links", {"app_name": "imagenet", "model_names": ["resnet-50"]}), calls)

    def test_register_idempotent_original_api(self):
        c = load()
        app = {"name": c["APPLICATION"], "input_type": "strings", "default_output": "CLIPPER_SLO_FALLBACK",
               "latency_slo_micros": 200000, "linked_models": ["resnet-50"]}
        model = {"model_name": "resnet-50", "model_version": "1"}  # batch_size not exposed
        calls = []

        def fake_http(url, payload):
            calls.append(url.split("/")[-1])
            if url.endswith("get_all_applications"):
                return [app]
            if url.endswith("get_all_models"):
                return [model]
            return "ok"

        with tempfile.TemporaryDirectory() as directory:
            active = Path(directory) / "active.json"
            active.write_text(json.dumps({"config": c}))
            with patch.object(client, "http_json", side_effect=fake_http), \
                 patch.object(deploy, "paths", return_value=(None, active, None)), \
                 contextlib.redirect_stdout(io.StringIO()):
                deploy.register(c)
        self.assertNotIn("add_model_links", calls)
        self.assertNotIn("add_app", calls)
        self.assertNotIn("add_model", calls)
        self.assertIn("set_model_version", calls)


if __name__ == "__main__":
    unittest.main()
