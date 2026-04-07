from act.detr.models.me_block.annotate_importance_labels import main


DEFAULT_DATA_ROOT = "./data_process/data"
DEFAULT_LABEL_DIRNAME = "importance_labels"


if __name__ == "__main__":
    main(default_data_root=DEFAULT_DATA_ROOT, default_label_dirname=DEFAULT_LABEL_DIRNAME)
