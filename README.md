# Using DreamerV3 to Train Freerouting Auto-Routing (JPype Method)

This project demonstrates how to use the `JPype` library to wrap an existing Java application (`freerouting.jar`) as a custom `gymnasium` reinforcement learning environment, and then train it using the **DreamerV3** framework.

---

## Project Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```
---

## Start Training (Description)

Once all project files (including the generated `freerouting_env.py`) are placed in the correct locations, you can start the DreamerV3 training process.

---

### 1. Open the Terminal

* Use **Command Prompt** or **PowerShell** on Windows, and **Terminal** on macOS/Linux.

---

### 2. Navigate to the Project Root Directory

* Use the `cd` command to switch to the main project folder (example: `dreamerv3-freerouting-env`):

```bash
cd path/to/your/projects/dreamer-Autorouting
```

* Make sure you run the training command from this directory so that both the DreamerV3 library and the custom environment module can be properly located.

---

### 3. Run the Training Command

```bash
python dreamerv3/train.py --configs defaults freerouting
```

**Command breakdown:**

* `python dreamerv3/train.py`: Executes the main DreamerV3 training script.
* `--configs`: Tells `train.py` to load one or more configuration sets.
* `defaults freerouting`: Loads the default parameters first, then overrides them with the custom `freerouting` configuration (e.g., using the `Freerouting-v0` environment).

---

### 4. Monitor the Training Process

* When training starts, logs will appear in the terminal (these may include JPype/JVM startup messages).
* Example expected messages:

  * “Creating Gym environment: Freerouting-v0”
  * Training steps, rewards, and loss updates.
* Training results will be saved in the project root directory under `logdir`, including:

  * Model weights
  * Evaluation videos
  * TensorBoard logs

---

### Notes

* Ensure that the API configurations of `freerouting.jar` and `freerouting_env.py`, as well as JPype startup parameters, are correctly set; otherwise, JVM initialization or environment creation errors may occur.
* If you encounter errors, verify your Python dependencies, JPype version, and JVM path settings.

---
