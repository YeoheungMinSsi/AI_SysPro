import kagglehub

# Download latest version
path = kagglehub.dataset_download("volkandl/car-brand-logos")

print("Path to dataset files:", path)