rel_train_data_path = "data/data_bb1_img/train"
rel_test_data_path = "data/data_bb1_img/test"
rel_model_path = "models"
rel_submission_path = "out/submission.json"

# model_type = "dummy"  # vggface2
model_type = "vggface2"  # vggface2
tasks = ["skin_tone", "gender", "age"]
img_size = 160
batch_size = 32
epochs = 5
# do_not_use_all_examples = True  # for debugging
do_not_use_all_examples = False  # for debugging

submission_title = "ky"
