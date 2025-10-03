import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno   
plt.rcParams['figure.figsize'] = (10,4)
sns.set_style("whitegrid")

# -------------

def quick_overview(df):
    print("----- Quick overview -----")
    print("Shape:", df.shape)
    print("\nTipe kolom:\n", df.dtypes)
    print("\nMissing counts per column:\n", df.isna().sum())
    print("\nContoh 5 baris:\n", df.head())

# -------------

num = df.select_dtypes(include=['number']).columns.tolist()
if not num:
    df[num] = df[num].apply(pd.to_numeric, errors='coerce')
if num:
    df[num].hist(bins=15, figsize=(12,3*max(1,len(num)//3)))
    plt.suptitle("Distribusi numeric (sebelum)"); plt.show()

# -------------

for c in num:
    sns.boxplot(x=df[c]); plt.title(c); plt.show()
cats = df.select_dtypes(include=['object','category']).columns.tolist()
for c in cats:
    df[c].value_counts(dropna=False).head(15).plot.bar(); plt.title(c); plt.show()

# -------------

def build_and_run_pipeline_short(X, y=None, test_size=0.2, random_state=42, out_dir="outputs"):
    X = X.copy()
    os.makedirs(out_dir, exist_ok=True)

    
    for c in X.columns:
        if X[c].dtype == 'object':
            conv = pd.to_numeric(X[c], errors='coerce')
            if conv.notna().sum() > 0.5 * len(conv):
                X[c] = conv

    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    
    if y is not None:
        strat = y if (y.nunique(dropna=True) > 1) else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)
    else:
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        y_train = y_test = None

  
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('scaler', StandardScaler())]) if num_cols else None
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))]) if cat_cols else None

    transformers = []
    if num_pipe: transformers.append(('num', num_pipe, num_cols))
    if cat_pipe: transformers.append(('cat', cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers, remainder='drop')

    
    X_train_pp = preprocessor.fit_transform(X_train)
    X_test_pp = preprocessor.transform(X_test)

  
    pd.DataFrame(X_train_pp).to_csv(os.path.join(out_dir, "X_train_preprocessed.csv"), index=False)
    pd.DataFrame(X_test_pp).to_csv(os.path.join(out_dir, "X_test_preprocessed.csv"), index=False)
    if y is not None:
        pd.Series(y_train).to_csv(os.path.join(out_dir, "y_train.csv"), index=False, header=False)
        pd.Series(y_test).to_csv(os.path.join(out_dir, "y_test.csv"), index=False, header=False)

    joblib.dump(preprocessor, os.path.join(out_dir, "preprocessor.joblib"))
    print("Preprocessing selesai. Files ->", out_dir)
    return preprocessor, (X_train_pp, X_test_pp, y_train, y_test, X_train, X_test)

# -------------

if __name__ == "__main__":
    DATA_PATH = "lung-cancer.data"  
    df = read_dataset(DATA_PATH, sep=",")
    quick_overview(df)

    X, y, target_col = infer_target(df)
    preprocessor, outputs = build_and_run_pipeline_short(X, y)

    X_train_pp, X_test_pp, y_train, y_test, X_train, X_test = outputs

    print("Shapes setelah preprocessing:")
    print("X_train:", X_train_pp.shape, "X_test:", X_test_pp.shape)
    if y is not None:
        print("y_train:", len(y_train), "y_test:", len(y_test))