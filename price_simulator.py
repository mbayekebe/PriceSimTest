import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

# Charger les données
data = pd.read_csv('train.csv')

# Sélectionner quelques caractéristiques pertinentes
features = ['GrLivArea', 'YearBuilt', 'TotalBsmtSF', 'FullBath', 'BedroomAbvGr']
X = data[features]
y = data['SalePrice']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error on Test Data: {mse:.2f}")

# Fonction de prédiction du prix de la maison
def predict_price(gr_liv_area, year_built, total_bsmt_sf, full_bath, bedroom_abv_gr):
    input_data = pd.DataFrame({
        'GrLivArea': [gr_liv_area],
        'YearBuilt': [year_built],
        'TotalBsmtSF': [total_bsmt_sf],
        'FullBath': [full_bath],
        'BedroomAbvGr': [bedroom_abv_gr]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Interface utilisateur avec Streamlit
st.title('Real Estate Price Estimator')

st.header('Input the characteristics of the house:')
gr_liv_area = st.number_input('Ground Living Area (in square feet)', min_value=500, max_value=5000, value=1500)
year_built = st.number_input('Year Built', min_value=1800, max_value=2022, value=2000)
total_bsmt_sf = st.number_input('Total Basement Area (in square feet)', min_value=0, max_value=3000, value=800)
full_bath = st.number_input('Number of Full Bathrooms', min_value=1, max_value=4, value=2)
bedroom_abv_gr = st.number_input('Number of Bedrooms Above Ground', min_value=1, max_value=8, value=3)

if st.button('Predict Price'):
    predicted_price = predict_price(gr_liv_area, year_built, total_bsmt_sf, full_bath, bedroom_abv_gr)
    st.subheader(f'Predicted Price: ${predicted_price:.2f}')
