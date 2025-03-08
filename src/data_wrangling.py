import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

# Función para detectar outlayers con el método IQR, extremos y no extremos
def iqr_method(column):
    """Detecta outliers usando el método IQR."""
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    extreme_lower_bound = q1 - 3 * iqr
    extreme_upper_bound = q3 + 3 * iqr
    return pd.Series([column>upper_bound, column<lower_bound, column>extreme_upper_bound, column<extreme_lower_bound], 
                     index=['outliers_upper', 'outliers_lower', 'extreme_upper', 'extreme_lower'])

# Función para realizar el preprocesamiento de los datos realizado con ayuda de Gemini 2.0 y claude sonnet 3.7
def data_wrangling(df):
    """Realiza el preprocesamiento de los datos."""

    # Filter and select columns from the original dataframe
    df_filtered = df[[
        'log_price','property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'city', 'neighbourhood', 'amenities',
        'review_scores_rating', 'number_of_reviews', 'cleaning_fee', 'bed_type',
        'cancellation_policy', 'instant_bookable', 'host_identity_verified',
        'host_response_rate', 'host_since', 'host_has_profile_pic'
    ]].copy()

    # Convert columns to appropriate data types
    df_filtered['log_price'] = df_filtered['log_price'].astype(float)
    df_filtered['accommodates'] = df_filtered['accommodates'].astype(int)
    df_filtered['bathrooms'] = df_filtered['bathrooms'].astype(float)
    df_filtered['bedrooms'] = df_filtered['bedrooms'].astype(float)
    df_filtered['beds'] = df_filtered['beds'].astype(float)
    df_filtered['review_scores_rating'] = df_filtered['review_scores_rating'].astype(float)
    df_filtered['number_of_reviews'] = df_filtered['number_of_reviews'].astype(int)

    # Convert categorical and boolean columns
    categorical_cols = ['property_type', 'room_type', 'city', 'bed_type', 'cancellation_policy']
    for col in categorical_cols:
        df_filtered[col] = df_filtered[col].astype('category')

    # Handle boolean columns that might have string values like 't' or 'f'
    bool_cols = ['instant_bookable', 'host_identity_verified', 'host_has_profile_pic']
    for col in bool_cols:
        if df_filtered[col].dtype == 'object':
            # Convert string representations like 't', 'f', 'true', 'false' to boolean
            df_filtered[col] = df_filtered[col].map(lambda x: 
                                            True if str(x).lower() in ['t', 'true', 'yes', '1'] 
                                            else False if str(x).lower() in ['f', 'false', 'no', '0'] 
                                            else x)
        df_filtered[col] = df_filtered[col].astype(bool)

    # Convert date column
    df_filtered['host_since'] = pd.to_datetime(df_filtered['host_since'])

    # Handle host_response_rate (remove % and convert to float)
    df_filtered['host_response_rate'] = df_filtered['host_response_rate'].str.replace('%', '').astype(float)

    # Imputar valores nulos en columnas de string con "Desconocido"
    df_filtered['neighbourhood'] = df_filtered['neighbourhood'].fillna('Desconocido')
    df_filtered['neighbourhood'] = df_filtered['neighbourhood'].astype('category')

    # Eliminar registros donde 'beds' es igual a 0 (inconsistente)
    df_filtered = df_filtered[df_filtered['beds'] != 0]

    # Redondear la columna 'bathrooms' a números enteros
    df_filtered['bathrooms'] = df_filtered['bathrooms'].round()

    # Impute zero bathrooms with 1
    df_filtered.loc[df_filtered['bathrooms'] == 0, 'bathrooms'] = 1

    # Filter extreme inconsistencies
    inconsistent_filter = (
        (df_filtered['accommodates'] >= 8) &
        (df_filtered['bathrooms'] <= 1) &
        (df_filtered['bedrooms'] <= 1) &
        (df_filtered['beds'] <= 1)
    )
    extreme_inconsistency = df_filtered[inconsistent_filter]
    df_filtered = df_filtered.drop(extreme_inconsistency.index)

    # Eliminar outliers inconsistentes en 'accommodates'
    outliers_accommodates = iqr_method(df_filtered['accommodates'])
    upper_outliers_accommodates = outliers_accommodates['outliers_upper']
    extreme_upper_accommodates = outliers_accommodates['extreme_upper']
    combined_outliers = upper_outliers_accommodates | extreme_upper_accommodates

    inconsistent_entries = df_filtered[combined_outliers][df_filtered[combined_outliers]['accommodates'] // 2 > df_filtered[combined_outliers]['beds']]
    df_filtered = df_filtered.drop(inconsistent_entries.index)

    # Imputar valores nulos en columnas numéricas con la media
    df_filtered['host_response_rate'] = df_filtered['host_response_rate'].fillna(df_filtered['host_response_rate'].mean())
    df_filtered['bathrooms'] = df_filtered['bathrooms'].fillna(round(df_filtered['bathrooms'].mean()))
    df_filtered['bedrooms'] = df_filtered['bedrooms'].fillna(round(df_filtered['bedrooms'].mean()))
    df_filtered['beds'] = df_filtered['beds'].fillna(round(df_filtered['accommodates']//2))
    df_filtered['review_scores_rating'] = df_filtered['review_scores_rating'].fillna(df_filtered['review_scores_rating'].mean())

    # Imputar valores nulos en columnas booleanas del host asumiendo que no tienen perfil verificado
    df_filtered['host_identity_verified'] = df_filtered['host_identity_verified'].fillna(False)
    df_filtered['host_has_profile_pic'] = df_filtered['host_has_profile_pic'].fillna(False)

    # Imputar valores nulos en 'host_since' con la fecha más reciente presente en el dataset
    most_recent_date = df_filtered['host_since'].dropna().max()
    df_filtered['host_since'] = df_filtered['host_since'].fillna(most_recent_date)

    # Imputar 'host_response_rate' a 0% para los hosts que no tenían valor y asumimos son nuevos
    df_filtered.loc[df_filtered['host_identity_verified'].isnull(), 'host_response_rate'] = 0
    
    # Convert columns to integers
    df_filtered['accommodates'] = df_filtered['accommodates'].astype(int)
    df_filtered['bathrooms'] = df_filtered['bathrooms'].astype(int)
    df_filtered['bedrooms'] = df_filtered['bedrooms'].astype(int)
    df_filtered['beds'] = df_filtered['beds'].astype(int)
    return df_filtered

if __name__ == '__main__':
    source_file = '../data/raw/Airbnb_Data.csv'
    df_raw = pd.read_csv(source_file, low_memory=False)

    df_cleaned = data_wrangling(df_raw.copy()) 

    # Guardar el dataset limpio
    df_cleaned.to_csv('../data/processed/Airbnb_Cleaned.csv', index=False)