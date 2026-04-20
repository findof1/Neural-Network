#include "ganUtils.hpp"

void discriminatorBackpropagation(GANNetwork &fullNetwork, Sample &sample)
{
  Network &discriminator = fullNetwork.discriminator;
  Layer &outputLayer = discriminator.layers.back();
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

  for (int i = static_cast<int>(discriminator.layers.size()) - 2; i >= 0; i--)
  {
    Layer &layer = discriminator.layers[i];
    Layer &nextLayer = discriminator.layers[i + 1];

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

  for (int i = 0; i < discriminator.layers.size(); i++)
  {
    Eigen::VectorXf prev = (i == 0) ? sample.inputs : discriminator.layers[i - 1].a;

    Layer &layer = discriminator.layers[i];

    layer.dW += layer.delta * prev.transpose();
    layer.db += layer.delta;
  }
}

Eigen::VectorXf getInputGradient(GANNetwork &fullNetwork, Sample &sample)
{
  Network &discriminator = fullNetwork.discriminator;
  Layer &outputLayer = discriminator.layers.back();
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

  for (int i = static_cast<int>(discriminator.layers.size()) - 2; i >= 0; i--)
  {
    Layer &layer = discriminator.layers[i];
    Layer &nextLayer = discriminator.layers[i + 1];

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

  return discriminator.layers[0].W.transpose() * discriminator.layers[0].delta;
}

void generatorBackpropagation(GANNetwork &fullNetwork, const Eigen::VectorXf &noise, const Eigen::VectorXf &discriminatorGradient)
{
  Network &generator = fullNetwork.generator;
  Layer &outputLayer = generator.layers.back();
  outputLayer.delta = discriminatorGradient;
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

  for (int i = static_cast<int>(generator.layers.size()) - 2; i >= 0; i--)
  {
    Layer &layer = generator.layers[i];
    Layer &nextLayer = generator.layers[i + 1];

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

  for (int i = 0; i < generator.layers.size(); i++)
  {
    Eigen::VectorXf prev = (i == 0) ? noise : generator.layers[i - 1].a;

    Layer &layer = generator.layers[i];

    layer.dW += layer.delta * prev.transpose();
    layer.db += layer.delta;
  }
}

float computeDiscriminatorLoss(const Eigen::VectorXf &realOut, const Eigen::VectorXf &fakeOut)
{
  float loss = 0.0f;

  for (int i = 0; i < realOut.size(); i++)
    loss += -log(std::max(realOut(i), 1e-7f));

  for (int i = 0; i < fakeOut.size(); i++)
    loss += -log(std::max(1.0f - fakeOut(i), 1e-7f));

  return loss;
}

float computeGeneratorLoss(const Eigen::VectorXf &fakeOut)
{
  float loss = 0.0f;

  for (int i = 0; i < fakeOut.size(); i++)
    loss += -log(std::max(fakeOut(i), 1e-7f));

  return loss;
}