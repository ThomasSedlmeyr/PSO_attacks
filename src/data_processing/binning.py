import pandas as pd

def clean_cardio_data(cardio_path, target_path):
    df = pd.read_csv(cardio_path, delimiter=';')
    df.drop(["id"], axis=1, inplace=True)
    df['age'] = df["age"].apply(lambda x: int(x / 356.25))
    df['age'] = df['age'].apply(lambda x: round(x / 5) * 5)
    df['height'] = df['height'].apply(lambda x: round(x / 10) * 10)
    df['weight'] = df['weight'].apply(lambda x: round(x / 10) * 10)
    df['ap_hi'] = df['ap_hi'].apply(lambda x: round(x / 10) * 10)

    df.to_csv(target_path + 'cardio_cat.csv', index=False)
    return df

