from act.detr.models.me_block.train_importance_model import main


DEFAULT_DATA_ROOT = "./data_process/data"
DEFAULT_SAVE_ROOT = "./log/me_block"


if __name__ == "__main__":
    main(default_data_root=DEFAULT_DATA_ROOT, default_save_root=DEFAULT_SAVE_ROOT)
