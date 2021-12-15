import pickle
import pandas

with open('../overcooked_ai_py/data/human_data/clean_train_trials.pickle', 'rb') as f:
    content = pickle.load(f)

    print(content)

    # print(content.columns)

    # print(min(content[content['layout_name'] == 'asymmetric_advantages']['cur_gameloop_total']))
    # print(max(content[content['layout_name'] == 'asymmetric_advantages']['cur_gameloop_total']))
    # content.to_csv("clean_train_trials.csv", mode='w', header=True)

with open('../overcooked_ai_py/data/human_data/clean_test_trials.pickle', 'rb') as f:
    content = pickle.load(f)
    print(content)
    # content.to_csv("clean_test_trials.csv", mode='w', header=True)


