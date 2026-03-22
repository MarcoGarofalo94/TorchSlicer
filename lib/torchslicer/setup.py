from setuptools import setup, find_packages

requirements = [
    'numpy',
]

extras = {
    'grpc': ['grpcio>=1.64.0', 'grpcio-tools>=1.64.0', 'protobuf'],
    'monitor': [
        'opentelemetry-api>=1.24.0',
        'opentelemetry-sdk>=1.24.0',
        'opentelemetry-exporter-otlp-proto-grpc>=1.24.0',
        # optional — auto-propagates trace context across gRPC calls:
        # 'opentelemetry-instrumentation-grpc>=0.45b0',
    ],
}

setup(
    name='torchslicer',
    version='0.2.0',
    install_requires=requirements,
    extras_require=extras,
    packages=find_packages(),
    package_data={'torchslicer': ['transport/grpc/**/*.proto']},
    author='Marco Garofalo',
    author_email='garofalomarco58@gmail.com',
    description='Split learning framework for PyTorch.',
    url='https://github.com/MarcoGarofalo94/TorchSlicer',
)
