# Preprocess weather data
python weather_preprocess.py data/weather/kyoto.csv
python weather_preprocess.py data/weather/liestal.csv
python weather_preprocess.py data/weather/washingtondc.csv
python weather_preprocess.py data/weather/vancouver.csv

# Preprocess blossom data
python blossom_preprocess.py data/blossoms/kyoto.csv
python blossom_preprocess.py data/blossoms/liestal.csv
python blossom_preprocess.py data/blossoms/washingtondc.csv

# Partition data
python partition.py

# Train model
python train.py --model_file models/my_model.pt --epochs 1200

# Evaluate model on the test set
python predict.py --test_set --model models/my_model.pt --output my_results

# Extrapolate weather data for future years
python weather_extrapolate.py --location kyoto --start_year 2023 --end_year 2032 --input_dir data/weather_cache --output_dir data/weather_extrapolate
python weather_extrapolate.py --location liestal --start_year 2023 --end_year 2032 --input_dir data/weather_cache --output_dir data/weather_extrapolate
python weather_extrapolate.py --location washingtondc --start_year 2023 --end_year 2032 --input_dir data/weather_cache --output_dir data/weather_extrapolate
python weather_extrapolate.py --location vancouver --start_year 2023 --end_year 2032 --input_dir data/weather_cache --output_dir data/weather_extrapolate

# Evaluate model to get final output (we use our presaved model for this step, but model could be changed to models/my_model.pt to use the newly trained model)
python predict.py --model models/submitted_best.pt --dataset data/weather_extrapolate --output results --final_format