import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_invoice_data():
    conn = sqlite3.connect("H:/ML Projects/Inventory Analysis Project/data/inventory.db")

    query = """
    WITH purchase_agg AS (
        select
            p.PONumber,
            count(distinct p.Brand) as total_brands,
            sum(p.Quantity) as total_item_quantity,
            sum(p.Dollars) as total_item_dollars,
            avg(julianday(p.ReceivingDate) - julianday(p.PODate)) as avg_receiving_delay
        from purchases p
        group by p.PONumber
    )
    select
        vi.PONumber,
        vi.Quantity as invoice_quantity,
        vi.Dollars as invoice_dollars,
        vi.Freight,
        (julianday(vi.invoiceDate) - julianday(vi.PODate)) as days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) as days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    from vendor_invoice vi
    left join purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_invoice_risk_label(row):
    # Invoice total mismatch with item-level total
    if(abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5):
        return 1

    # High receiving delay
    if row["avg_receiving_delay"] > 10:
        return 1

    return 0

def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis = 1)
    return df

def split_data(df, features, target):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size = 0.2, random_state = 42)

def scale_features(X_train, X_test, scaler_path):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'models/scaler.pkl')
    return X_train_scaled, X_test_scaled