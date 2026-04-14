#include "neuralNetworkUtils.hpp"

void forwardPass(Network &net, const Eigen::VectorXf &input)
{
  Eigen::VectorXf current = input;

  for (auto &layer : net.layers)
  {
    layer.z = layer.W * current + layer.b;

    if (layer.activation == Sigmoid)
    {
      layer.a = 1.0f / (1.0f + (-layer.z.array()).exp());
    }
    else if (layer.activation == ReLU)
    {
      layer.a = layer.z.cwiseMax(0);
    }
    else if (&layer == &net.layers.back() && layer.activation == SoftMax)
    {
      float maxVal = layer.z.maxCoeff();
      Eigen::VectorXf shifted = layer.z.array() - maxVal;
      Eigen::VectorXf expVals = shifted.array().exp();
      layer.a = expVals / expVals.sum();
    }
    else
    {
      layer.a = layer.z;
    }

    current = layer.a;
  }
}

float computeCostMSE(Network &net, const Sample &sample)
{
  return (net.layers.back().a - sample.targets).array().square().mean();
}

float computeCostCrossEntropy(Network &net, const Sample &sample)
{
  Eigen::VectorXf &p = net.layers.back().a;

  float eps = 1e-8f;
  float loss = 0.0f;

  for (int i = 0; i < p.size(); i++)
  {
    loss -= sample.targets[i] * std::log(p[i] + eps);
  }

  return loss;
}

void backpropagation(Network &net, const Sample &sample)
{
  Layer &outputLayer = net.layers.back();
  outputLayer.delta = outputLayer.a - sample.targets;

  if (outputLayer.activation == Sigmoid)
  {
    outputLayer.delta =
        outputLayer.delta.array() *
        (outputLayer.a.array() * (1.0f - outputLayer.a.array()));
  }
  else if (outputLayer.activation == ReLU)
  {
    outputLayer.delta =
        outputLayer.delta.array() *
        (outputLayer.z.array() > 0).cast<float>();
  }

  for (int i = static_cast<int>(net.layers.size()) - 2; i >= 0; i--)
  {
    Layer &layer = net.layers[i];
    Layer &nextLayer = net.layers[i + 1];

    layer.delta = nextLayer.W.transpose() * nextLayer.delta;

    if (layer.activation == Sigmoid)
    {
      layer.delta =
          layer.delta.array() *
          (layer.a.array() * (1.0f - layer.a.array()));
    }
    else if (layer.activation == ReLU)
    {
      layer.delta =
          layer.delta.array() *
          (layer.z.array() > 0).cast<float>();
    }
  }

  for (int i = 0; i < net.layers.size(); i++)
  {
    Eigen::VectorXf prev =
        (i == 0) ? sample.inputs : net.layers[i - 1].a;

    Layer &layer = net.layers[i];

    layer.dW += layer.delta * prev.transpose();
    layer.db += layer.delta;
  }
}

void applyGradients(Network &net)
{
  float lr = net.config.learningRate;
  float scale = lr / net.config.batchSize;

  for (auto &layer : net.layers)
  {
    layer.W -= scale * layer.dW;
    layer.b -= scale * layer.db;

    layer.dW.setZero();
    layer.db.setZero();
  }
}

void addLayer(Network &net, int inputSize, int outputSize, LayerActivationType activation)
{
  Layer layer;

  std::random_device rd;
  std::mt19937 gen(rd());

  float stddev = 1.0f;

  if (activation == ReLU)
    stddev = std::sqrt(2.0f / inputSize);
  else if (activation == Sigmoid)
    stddev = std::sqrt(1.0f / inputSize);

  std::normal_distribution<float> dist(0.0f, stddev);

  layer.W = Eigen::MatrixXf(outputSize, inputSize);
  layer.b = Eigen::VectorXf::Zero(outputSize);

  for (int i = 0; i < outputSize; i++)
  {
    for (int j = 0; j < inputSize; j++)
    {
      layer.W(i, j) = dist(gen);
    }
  }

  layer.dW = Eigen::MatrixXf::Zero(outputSize, inputSize);
  layer.db = Eigen::VectorXf::Zero(outputSize);
  layer.z = Eigen::VectorXf(outputSize);
  layer.a = Eigen::VectorXf(outputSize);
  layer.delta = Eigen::VectorXf(outputSize);
  layer.activation = activation;

  net.layers.push_back(layer);
}

Dataset loadMNISTCSV(const std::string &path)
{
  Dataset dataset;
  dataset.samples.reserve(60000);

  std::ifstream file(path);
  std::string line;

  while (std::getline(file, line))
  {
    if (line.empty())
      continue;

    const char *ptr = line.c_str();
    char *end = nullptr;

    int label = std::strtol(ptr, &end, 10);

    if (ptr == end)
      continue;

    ptr = end;
    if (*ptr == ',')
      ptr++;

    Sample sample;
    sample.inputs = Eigen::VectorXf(784);
    sample.targets = Eigen::VectorXf::Zero(10);
    sample.targets[label] = 1.0f;

    for (int i = 0; i < 784; i++)
    {
      float pixel = std::strtof(ptr, &end);
      sample.inputs[i] = pixel / 255.0f;

      ptr = end;
      if (*ptr == ',')
        ptr++;
    }

    // sample.targets = sample.inputs; //for if I want to do image reconstruction
    dataset.samples.push_back(std::move(sample));
  }

  return dataset;
}

int predict(Network &net, const Eigen::VectorXf &input)
{
  forwardPass(net, input);

  Eigen::VectorXf &out = net.layers.back().a;

  int bestIndex = 0;
  float bestValue = out[0];

  for (int i = 1; i < out.size(); i++)
  {
    if (out[i] > bestValue)
    {
      bestValue = out[i];
      bestIndex = i;
    }
  }

  return bestIndex;
}

float computeAccuracy(Network &net, const Dataset &dataset)
{
  int correct = 0;

  for (const auto &sample : dataset.samples)
  {
    int predicted = predict(net, sample.inputs);

    int actual = 0;
    for (int i = 0; i < sample.targets.size(); i++)
    {
      if (sample.targets[i] == 1.0f)
      {
        actual = i;
        break;
      }
    }

    if (predicted == actual)
      correct++;
  }

  return static_cast<float>(correct) / dataset.samples.size();
}

QImage vectorToImage(const Eigen::VectorXf &v)
{
  QImage img(28, 28, QImage::Format_Grayscale8);

  for (int y = 0; y < 28; y++)
  {
    for (int x = 0; x < 28; x++)
    {
      int i = y * 28 + x;

      int value = static_cast<int>(v[i] * 255.0f);
      value = std::clamp(value, 0, 255);

      img.setPixel(x, y, qRgb(value, value, value));
    }
  }

  return img;
}