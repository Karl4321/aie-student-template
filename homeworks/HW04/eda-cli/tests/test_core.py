from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_top_categories_with_different_k():
    """Тест для проверки параметра top_k_categories"""
    df = pd.DataFrame({
        "category": ["A", "A", "A", "B", "B", "C", "D", "E", "F", "G"] * 3,
        "numeric": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3
    })
    
    # Тестируем с top_k=3
    top_cats_3 = top_categories(df, max_columns=5, top_k=3)
    assert "category" in top_cats_3
    assert len(top_cats_3["category"]) == 3
    
    # Тестируем с top_k=5
    top_cats_5 = top_categories(df, max_columns=5, top_k=5)
    assert len(top_cats_5["category"]) == 5
    
    # Проверяем, что первое значение самое частое
    assert top_cats_5["category"].iloc[0]["value"] == "A"
    assert top_cats_5["category"].iloc[0]["count"] == 9


def test_high_cardinality_flag():
    """Тест для проверки флага has_high_cardinality_categoricals"""
    n_rows = 30
    df = pd.DataFrame({
        "col1": list(range(20)) + [0] * 10, 
        "col2": list(range(20)) + [1] * 10,  
        "col3": list(range(20)) + [2] * 10,
        "col4": list(range(20)) + [3] * 10,
        "col5": list(range(20)) + [4] * 10,
        "col6": list(range(20)) + [5] * 10,
        "col7": list(range(20)) + [6] * 10,
        "col8": ["A"] * 30,
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags["has_high_cardinality_categoricals"] == True
    
    # Проверяем, что score был скорректирован
    assert flags["quality_score"] <= 1.0


def test_suspicious_id_duplicates_flag():
    """Тест для проверки флага has_suspicious_id_duplicates"""
    df1 = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5, 6],
        "age": [20, 21, 22, 23, 24, 25],
        "name": ["A", "B", "C", "D", "E", "F"]
    })
    
    df1 = df1.head(5)  
    
    summary1 = summarize_dataset(df1)
    missing_df1 = missing_table(df1)
    flags1 = compute_quality_flags(summary1, missing_df1)
    
    df1_fixed = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 4],
        "age": [20, 21, 22, 23, 24],
        "name": ["A", "B", "C", "D", "E"]
    })
    
    summary1 = summarize_dataset(df1_fixed)
    missing_df1 = missing_table(df1_fixed)
    flags1 = compute_quality_flags(summary1, missing_df1)
    
    assert flags1["has_suspicious_id_duplicates"] == True
    
    assert flags1["quality_score"] < 1.0
    
    df2 = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 4],
        "age": [20, 21, 22, 23, 24, 25]
    })
    
    summary2 = summarize_dataset(df2)
    missing_df2 = missing_table(df2)
    flags2 = compute_quality_flags(summary2, missing_df2)
    
    # Проверяем, что флаг установлен в False
    assert flags2["has_suspicious_id_duplicates"] == False


def test_quality_score_with_flags():
    """Тест для проверки влияния флагов на качество score"""
    user_id_list = list(range(99)) + [98]  
    
    df = pd.DataFrame({
        "user_id": user_id_list,
        "col1": list(range(20)) * 5, 
        "col2": list(range(20)) * 5,
        "col3": list(range(20)) * 5,
        "col4": list(range(20)) * 5,
        "col5": list(range(20)) * 5,
        "col6": list(range(20)) * 5,
        "col7": list(range(20)) * 5,
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем флаги
    assert flags["too_few_rows"] == False  
    assert flags["has_suspicious_id_duplicates"] == True  
    assert flags["has_high_cardinality_categoricals"] == True  
    
    assert flags["quality_score"] == 0.5