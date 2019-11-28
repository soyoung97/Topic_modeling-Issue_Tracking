import os
import json
import pandas
import LDA


from glob import glob


def get_articles(data_path):
    total_df = None
    for fname in glob(os.path.join(data_path, '*.json')):
        with open(fname, 'r') as f:
            data = json.load(f)
            temp_df = pandas.DataFrame.from_dict(data)

            if total_df is None:
                total_df = temp_df
            else:
                total_df = total_df.append(temp_df, ignore_index=True)

    return total_df


def main():
    df = get_articles('./data')
    topics = LDA.make_LDA_model(LDA.ALPHA).show_topics(num_topics=LDA.NUM_TOPICS,
                                                       num_words=LDA.NUM_WORDS)
    print(topics)


if __name__ == '__main__':
    main()
