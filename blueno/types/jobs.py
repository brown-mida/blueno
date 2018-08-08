import elasticsearch_dsl

TRAINING_JOBS = 'training_jobs'
VALIDATION_JOBS = 'validation_jobs'
JOB_INDEX = elasticsearch_dsl.Index(TRAINING_JOBS)
VALIDATION_JOB_INDEX = elasticsearch_dsl.Index(VALIDATION_JOBS)


class TrainingJob(elasticsearch_dsl.Document):
    id = elasticsearch_dsl.Integer()
    schema_version = elasticsearch_dsl.Integer()
    job_name = elasticsearch_dsl.Keyword()
    author = elasticsearch_dsl.Keyword()
    created_at = elasticsearch_dsl.Date()
    ended_at = elasticsearch_dsl.Date()
    params = elasticsearch_dsl.Text()
    raw_log = elasticsearch_dsl.Text()
    model_url = elasticsearch_dsl.Text()

    # Metrics
    epochs = elasticsearch_dsl.Integer()
    train_acc = elasticsearch_dsl.Float()
    final_val_acc = elasticsearch_dsl.Float()
    best_val_acc = elasticsearch_dsl.Float()
    final_val_loss = elasticsearch_dsl.Float()
    best_val_loss = elasticsearch_dsl.Float()
    final_val_sensitivity = elasticsearch_dsl.Float()
    best_val_sensitivity = elasticsearch_dsl.Float()
    final_val_specificity = elasticsearch_dsl.Float()
    best_val_specificity = elasticsearch_dsl.Float()
    final_val_auc = elasticsearch_dsl.Float()
    best_val_auc = elasticsearch_dsl.Float()

    # Params
    batch_size = elasticsearch_dsl.Integer()
    val_split = elasticsearch_dsl.Float()
    seed = elasticsearch_dsl.Integer()

    rotation_range = elasticsearch_dsl.Float()
    width_shift_range = elasticsearch_dsl.Float()
    height_shift_range: float = elasticsearch_dsl.Float()
    shear_range = elasticsearch_dsl.Float()
    zoom_range = elasticsearch_dsl.Keyword()
    horizontal_flip = elasticsearch_dsl.Boolean()
    vertical_flip = elasticsearch_dsl.Boolean()

    dropout_rate1 = elasticsearch_dsl.Float()
    dropout_rate2 = elasticsearch_dsl.Float()

    data_dir = elasticsearch_dsl.Keyword()
    gcs_url = elasticsearch_dsl.Keyword()

    mip_thickness = elasticsearch_dsl.Integer()
    height_offset = elasticsearch_dsl.Integer()
    pixel_value_range = elasticsearch_dsl.Keyword()

    # We need to keep a list of params for the parser because
    # we can't use traditional approaches to get the class attrs
    params_to_parse = ['batch_size',
                       'val_split',
                       'seed',
                       'rotation_range',
                       'width_shift_range',
                       'height_shift_range',
                       'shear_range',
                       'zoom_range',
                       'horizontal_flip',
                       'vertical_flip',
                       'dropout_rate1',
                       'dropout_rate2',
                       'data_dir',
                       'gcs_url',
                       'mip_thickness',
                       'height_offset',
                       'pixel_value_range']

    class Index:
        name = TRAINING_JOBS


class ValidationJob(elasticsearch_dsl.Document):
    """
    Object for validation data.
    TODO: Can this be merged with TrainingJob, with a common
        parent object?
    """
    id = elasticsearch_dsl.Integer()
    schema_version = elasticsearch_dsl.Integer()
    job_name = elasticsearch_dsl.Keyword()
    author = elasticsearch_dsl.Keyword()
    created_at = elasticsearch_dsl.Date()
    params = elasticsearch_dsl.Text()
    raw_log = elasticsearch_dsl.Text()

    # Metrics
    purported_acc = elasticsearch_dsl.Float()
    purported_loss = elasticsearch_dsl.Float()
    purported_sensitivity = elasticsearch_dsl.Float()

    avg_test_acc = elasticsearch_dsl.Float()
    avg_test_loss = elasticsearch_dsl.Float()
    avg_test_sensitivity = elasticsearch_dsl.Float()
    avg_test_specificity = elasticsearch_dsl.Float()
    avg_test_true_pos = elasticsearch_dsl.Float()
    avg_test_false_neg = elasticsearch_dsl.Float()
    avg_test_auc = elasticsearch_dsl.Float()

    best_test_acc = elasticsearch_dsl.Float()
    best_test_loss = elasticsearch_dsl.Float()
    best_test_sensitivity = elasticsearch_dsl.Float()
    best_test_specificity = elasticsearch_dsl.Float()
    best_test_true_pos = elasticsearch_dsl.Float()
    best_test_false_neg = elasticsearch_dsl.Float()
    best_test_auc = elasticsearch_dsl.Float()
    best_end_val_acc = elasticsearch_dsl.Float()
    best_end_val_loss = elasticsearch_dsl.Float()
    best_max_val_acc = elasticsearch_dsl.Float()
    best_max_val_loss = elasticsearch_dsl.Float()

    # Params
    batch_size = elasticsearch_dsl.Integer()
    val_split = elasticsearch_dsl.Float()
    seed = elasticsearch_dsl.Integer()

    rotation_range = elasticsearch_dsl.Float()
    width_shift_range = elasticsearch_dsl.Float()
    height_shift_range = elasticsearch_dsl.Float()
    shear_range = elasticsearch_dsl.Float()
    zoom_range = elasticsearch_dsl.Keyword()
    horizontal_flip = elasticsearch_dsl.Boolean()
    vertical_flip = elasticsearch_dsl.Boolean()

    dropout_rate1 = elasticsearch_dsl.Float()
    dropout_rate2 = elasticsearch_dsl.Float()

    data_dir = elasticsearch_dsl.Keyword()
    gcs_url = elasticsearch_dsl.Keyword()

    mip_thickness = elasticsearch_dsl.Integer()
    height_offset = elasticsearch_dsl.Integer()
    pixel_value_range = elasticsearch_dsl.Keyword()

    # We need to keep a list of params for the parser because
    # we can't use traditional approaches to get the class attrs
    params_to_parse = ['batch_size',
                       'val_split',
                       'seed',
                       'rotation_range',
                       'width_shift_range',
                       'height_shift_range',
                       'shear_range',
                       'zoom_range',
                       'horizontal_flip',
                       'vertical_flip',
                       'dropout_rate1',
                       'dropout_rate2',
                       'data_dir',
                       'gcs_url',
                       'mip_thickness',
                       'height_offset',
                       'pixel_value_range']

    class Index:
        name = VALIDATION_JOBS
