import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame):

    # 1. Tangani missing value (nilai 0 -> median)
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero:
        df[col] = df[col].replace(0, df[col].median())

    # 2. Hapus data duplikat
    df = df.drop_duplicates()

    # 3. Standarisasi fitur numerik (kecuali target)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop('Outcome', axis=1))

    # Gabungkan hasil scaling dengan target kembali
    X_scaled_df = pd.DataFrame(X_scaled, columns=df.drop('Outcome', axis=1).columns)
    final_df = pd.concat([X_scaled_df, df['Outcome'].reset_index(drop=True)], axis=1)

    return final_df


if __name__ == "__main__":
    # eksekusi langsung dari file
    file_path = "../dataset_raw/diabetes.csv"   
    df = pd.read_csv(file_path)

    processed_df = preprocess_data(df)

    # tampilkan ringkasan hasil
    print("Shape data setelah preprocessing:", processed_df.shape)
    print(processed_df.head())

    # simpan ke file baru
    processed_df.to_csv("dataset_preprocessing/diabetes_preprocessed.csv", index=False)
    print("Dataset hasil preprocessing disimpan ke diabetes_preprocessed.csv")
