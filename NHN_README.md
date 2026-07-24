# Board-Insertion Robometer LoRA Fine-Tuning

Run all commands from the Robometer repository root.

## 1. Prepare the processed cache

The Hugging Face train and test datasets must already exist as:

- `chomeed/board_insertion_train_rfm`
- `chomeed/board_insertion_test_rfm`

Authenticate if these repositories are private:

```bash
export HF_TOKEN="your_huggingface_token"
```

Download the video files referenced by the Hugging Face dataset tables:

```bash
export ROBOMETER_DATASET_PATH="$PWD/robometer_dataset"
mkdir -p "$ROBOMETER_DATASET_PATH"

hf download chomeed/board_insertion_train_rfm \
  --repo-type dataset \
  --include "board_insertion_train_rfm/**" \
  --local-dir "$ROBOMETER_DATASET_PATH"

hf download chomeed/board_insertion_test_rfm \
  --repo-type dataset \
  --include "board_insertion_test_rfm/**" \
  --local-dir "$ROBOMETER_DATASET_PATH"
```

The download root must contain:

```text
$ROBOMETER_DATASET_PATH/
├── board_insertion_train_rfm/
└── board_insertion_test_rfm/
```

Preprocess both datasets into the cache consumed by the training dataloader:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python robometer/data/scripts/preprocess_datasets.py \
  --config_path robometer/configs/preprocess_board_insertion.yaml
```

Point Robometer at the resulting cache:

```bash
export ROBOMETER_PROCESSED_DATASETS_PATH="$PWD/processed_datasets/board_insertion_rfm"
```

`ROBOMETER_DATASET_PATH` is the input-video location used during
preprocessing. `ROBOMETER_PROCESSED_DATASETS_PATH` is the generated NPZ/cache
location used during training. They are different directories.

The expected cache directories are:

```text
chomeed_board_insertion_train_rfm_board_insertion_train_rfm
chomeed_board_insertion_test_rfm_board_insertion_test_rfm
```

## 2. Run LoRA fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=true \
  model.train_progress_head=true \
  model.train_preference_head=true \
  training.load_from_checkpoint=robometer/Robometer-4B \
  data.train_datasets=[chomeed_board_insertion_train_rfm_board_insertion_train_rfm] \
  data.eval_datasets=[chomeed_board_insertion_test_rfm_board_insertion_test_rfm] \
  training.per_device_train_batch_size=8 \
  training.per_device_eval_batch_size=8 \
  training.gradient_accumulation_steps=1 \
  training.learning_rate=2e-5 \
  training.warmup_ratio=0.1 \
  training.weight_decay=0.01 \
  training.max_steps=1000 \
  training.output_dir=./logs \
  training.exp_name=board_insertion_lora \
  training.overwrite_output_dir=true \
  training.evaluation_strategy=steps \
  training.do_eval=true \
  training.run_default_eval=false \
  training.eval_steps=100 \
  training.custom_eval_steps=100 \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=[chomeed_board_insertion_test_rfm_board_insertion_test_rfm] \
  custom_eval.policy_ranking=[chomeed_board_insertion_test_rfm_board_insertion_test_rfm] \
  custom_eval.reward_alignment_max_trajectories=10 \
  custom_eval.policy_ranking_max_tasks=1 \
  logging.log_to=[wandb] \
  logging.wandb_project=robometer-board-insertion
```

The training split supplies gradients. The held-out test split is used for
reward-alignment and policy-ranking evaluation only.

## 3. Memory adjustment

If batch size 8 causes an out-of-memory error, preserve the effective batch
size by reducing the per-device batch size and increasing gradient
accumulation:

```bash
training.per_device_train_batch_size=2 \
training.gradient_accumulation_steps=4
```

Evaluation is intentionally run every 100 steps because the custom evaluations
are more expensive than the training step. Reduce the interval only when more
frequent validation is worth the additional runtime.
