Aplikasi ini memprediksi jumlah perjalanan taksi (trips) di New York City berdasarkan lokasi penjemputan (PULocationID), jam pengambilan (pickup_hour), dan hari dalam minggu (pickup_dayofweek).
Model dibangun menggunakan pendekatan machine learning dengan eksplorasi pola musiman (seasonality), korelasi antar fitur, serta optimisasi model terbaik menggunakan Hyperparameter Tuning.


### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
