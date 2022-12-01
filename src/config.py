rel_train_data_path = "../data/train"
rel_test_data_path = "../data/test"
rel_model_path = "../models"
rel_submission_path = "../out/submission.json"

model_type = "dummy"  # vggface2
tasks = ["skin_tone", "gender", "age"]
img_size = 128
batch_size = 32
epochs = 5
do_not_use_all_examples = True  # for debugging

submission_title = "ky"
