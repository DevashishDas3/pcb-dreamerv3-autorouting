import hashlib
import os
import warnings

import gym
import numpy as np

try:
    import jpype
except Exception:
    jpype = None


class FreeroutingJPypeEnv(gym.Env):
    """Freerouting environment with legacy and newer JPype backend support."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        jar_path="",
        dsn_file_path="",
        seed=0,
        action_size=100,
        obs_shape=(64, 64, 3),
        max_steps=200,
    ):
        super().__init__()
        self._rng = np.random.RandomState(seed)
        self._max_steps = max_steps
        self._step_count = 0
        self._dummy_mode = True
        self._backend = "dummy"
        self._controller = None
        self._status_controller = None
        self._session_controller = None
        self._job_controller = None
        self._dsn_file_path = dsn_file_path

        self.action_space = gym.spaces.Discrete(action_size)
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=obs_shape,
                    dtype=np.uint8,
                )
            }
        )

        print("Freerouting backend: initializing")

        if not jpype:
            warnings.warn("JPype not installed; using dummy freerouting environment.")
            print("Freerouting backend: dummy")
            return

        if not (
            jar_path
            and dsn_file_path
            and os.path.exists(jar_path)
            and os.path.exists(dsn_file_path)
        ):
            warnings.warn(
                "Freerouting jar or dsn path is missing/invalid; using dummy freerouting environment."
            )
            print("Freerouting backend: dummy")
            return

        try:
            if not jpype.isJVMStarted():
                jpype.startJVM(classpath=[jar_path])
        except Exception as err:
            warnings.warn(
                f"Failed to start JVM for freerouting ({err}); using dummy environment."
            )
            print("Freerouting backend: dummy")
            return

        try:
            if self._class_exists("app.freerouting.logic.RoutingController"):
                self._init_legacy_backend()
            elif self._class_exists("app.freerouting.api.v1.SystemControllerV1"):
                self._init_v1_backend()
            else:
                warnings.warn(
                    "No known freerouting JPype backend class found in jar; using dummy environment."
                )
                print("Freerouting backend: dummy")
        except Exception as err:
            warnings.warn(
                f"Failed to initialize freerouting JPype backend ({err}); using dummy environment."
            )
            print("Freerouting backend: dummy")

    def _class_exists(self, class_name):
        try:
            jpype.JClass(class_name)
            return True
        except Exception as err:
            if "UnsupportedClassVersionError" in str(err):
                warnings.warn(
                    "Freerouting jar needs a newer Java runtime. "
                    "Install a newer JDK and rerun to enable JPype backend."
                )
            return False

    def _init_legacy_backend(self):
        routing_controller = jpype.JClass("app.freerouting.logic.RoutingController")
        self._controller = routing_controller(self._dsn_file_path)
        self._dummy_mode = False
        self._backend = "jpype_legacy"
        print("Freerouting backend: JPype legacy RoutingController")

    def _init_v1_backend(self):
        self._status_controller = jpype.JClass(
            "app.freerouting.api.v1.SystemControllerV1"
        )()
        self._session_controller = jpype.JClass(
            "app.freerouting.api.v1.SessionControllerV1"
        )()
        self._job_controller = jpype.JClass("app.freerouting.api.v1.JobControllerV1")()
        _ = self._response_status(self._status_controller.getStatus())
        self._dummy_mode = False
        self._backend = "jpype_api_v1"
        print("Freerouting backend: JPype API v1 controllers")

    def _dummy_obs(self):
        image = self._rng.randint(
            0,
            256,
            size=self.observation_space["image"].shape,
            dtype=np.uint8,
        )
        return {"image": image}

    def _text_obs(self, text):
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        seed = int.from_bytes(digest[:4], byteorder="little", signed=False)
        rng = np.random.RandomState(seed)
        image = rng.randint(
            0,
            256,
            size=self.observation_space["image"].shape,
            dtype=np.uint8,
        )
        return {"image": image}

    def _response_status(self, response):
        try:
            return int(response.getStatus())
        except Exception:
            return -1

    def _response_entity_text(self, response):
        try:
            entity = response.getEntity()
            if entity is None:
                return str(response)
            return str(entity)
        except Exception:
            return str(response)

    def reset(self):
        self._step_count = 0
        if self._dummy_mode:
            obs = self._dummy_obs()
            obs["is_first"] = True
            obs["is_terminal"] = False
            return obs

        if self._backend == "jpype_legacy":
            self._controller.reset()
            java_observation = self._controller.getObservation()
            image = np.asarray(java_observation, dtype=np.uint8)
            return {"image": image, "is_first": True, "is_terminal": False}

        if self._backend == "jpype_api_v1":
            try:
                health = self._status_controller.getStatus()
                obs = self._text_obs(self._response_entity_text(health))
            except Exception as err:
                warnings.warn(
                    f"API v1 reset failed ({err}); switching to dummy observations."
                )
                self._dummy_mode = True
                self._backend = "dummy"
                obs = self._dummy_obs()
            obs["is_first"] = True
            obs["is_terminal"] = False
            return obs

        obs = self._dummy_obs()
        obs["is_first"] = True
        obs["is_terminal"] = False
        return obs

    def step(self, action):
        self._step_count += 1

        if self._dummy_mode:
            obs = self._dummy_obs()
            obs["is_first"] = False
            done = self._step_count >= self._max_steps
            obs["is_terminal"] = bool(done)
            return obs, 0.0, done, {"backend": "dummy"}

        if self._backend == "jpype_legacy":
            reward = float(self._controller.performAction(int(action)))
            java_observation = self._controller.getObservation()
            image = np.asarray(java_observation, dtype=np.uint8)
            done = bool(self._controller.isFinished())
            return (
                {
                    "image": image,
                    "is_first": False,
                    "is_terminal": done,
                },
                reward,
                done,
                {"backend": self._backend},
            )

        if self._backend == "jpype_api_v1":
            done = self._step_count >= self._max_steps
            reward = 0.0
            try:
                import sys
                import time

                t0 = time.time()
                health = self._status_controller.getStatus()
                t1 = time.time()
                if t1 - t0 > 0.1:
                    print(
                        f"[SLOW] getStatus took {t1-t0:.3f}s",
                        flush=True,
                        file=sys.stderr,
                    )
                status_code = self._response_status(health)
                text = self._response_entity_text(health)
                obs = self._text_obs(f"{status_code}:{text}:{int(action)}")
                if status_code >= 500:
                    reward = -1.0
                if self._step_count % 100 == 0:
                    print(
                        f"[V1] step {self._step_count} ok", flush=True, file=sys.stderr
                    )
            except Exception as err:
                warnings.warn(
                    f"API v1 step failed ({err}); falling back to dummy mode."
                )
                self._dummy_mode = True
                self._backend = "dummy"
                obs = self._dummy_obs()

            obs["is_first"] = False
            obs["is_terminal"] = bool(done)
            return obs, reward, done, {"backend": "jpype_api_v1"}

        obs = self._dummy_obs()
        obs["is_first"] = False
        done = self._step_count >= self._max_steps
        obs["is_terminal"] = bool(done)
        return obs, 0.0, done, {"backend": "dummy"}

    def close(self):
        return None
