from core.data_generator import generate_synthetic_data
from core.modeling import train_all

if __name__ == "__main__":
    df=generate_synthetic_data(n_skus=12,n_days=120,save=False)
    assert len(df)>0
    assert {0,1}.issuperset(set(df["Stockout_Next_7D"].unique()))
    print("rows",len(df),"stockout_rate",df["Stockout_Next_7D"].mean())
    print("Synthetic data smoke test passed")
