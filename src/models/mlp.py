import torch as th

# Is cuda available and move model to cuda if it is
if th.cuda.is_available():

    # Print the GPU name
    gpu_name = th.cuda.get_device_name(0)
    print(f"Using GPU: {gpu_name}")

else:

    print("CUDA is not available. Using CPU.")
    device = th.device("cpu")

model = th.nn.Sequential(
    th.nn.Linear(784, 256),
    th.nn.ReLU(),
    th.nn.Linear(256, 128),
    th.nn.ReLU(),
    th.nn.Linear(128, 64),
    th.nn.ReLU(),
    th.nn.Linear(64, 10),
    th.nn.Softmax(dim=1)
)

# Example network usage
input = th.randn(1, 784)
output = model(input)
print("Model output:")
print(output)
