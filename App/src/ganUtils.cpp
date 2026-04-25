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

Eigen::VectorXf computeMeanImage(const Dataset &data, int digit)
{
  int imageVectorSize = data.samples[0].inputs.size(); // 784 for MNIST becuase 28*28 images are inputs
  Eigen::VectorXf meanImage = Eigen::VectorXf::Zero(imageVectorSize);
  int digitCount = 0;
  for (const Sample &s : data.samples)
  {
    if (s.digit == digit)
    {
      meanImage += s.inputs;
      digitCount++;
    }
  }

  if (digitCount > 0)
  {
    meanImage /= digitCount;
  }
  else
  {
    std::cout << "No samples found for this digit when calculating mean image: " << digit << std::endl;
  }

  return meanImage;
}

Eigen::VectorXf computeStdDevImage(const Dataset &data, const Eigen::VectorXf &meanImage, int digit)
{
  int imageVectorSize = meanImage.size();
  Eigen::VectorXf stdDevImage = Eigen::VectorXf::Zero(imageVectorSize);
  int digitCount = 0;
  for (const Sample &s : data.samples)
  {
    if (s.digit == digit)
    {
      Eigen::VectorXf diff = s.inputs - meanImage;
      stdDevImage += diff.array().square().matrix();
      digitCount++;
    }
  }

  if (digitCount > 0)
  {
    stdDevImage /= digitCount;
  }
  else
  {
    std::cout << "No samples found for this digit when calculating std dev image: " << digit << std::endl;
  }

  stdDevImage = stdDevImage.array().sqrt();

  return stdDevImage;
}

Eigen::VectorXf computeDatasetVariance(Dataset &data)
{
  Eigen::VectorXf f_divs = Eigen::VectorXf::Zero(10); // for digits 0,1,2,3,4,5,6,7,8,9
  // loops over all different sample varieties we want to calculate f_div(digit) for
  for (int digit = 0; digit <= 9; digit++)
  {
    const Eigen::VectorXf meanImage = computeMeanImage(data, digit);
    Eigen::VectorXf stdDevImage = computeStdDevImage(data, meanImage, digit);
    float f_div = stdDevImage.mean();
    f_divs[digit] = f_div;
  }

  float maxVal = f_divs.maxCoeff();
  if (maxVal > 0.0f)
  {
    f_divs /= maxVal;
  }
  else
  {
    std::cout << "Max variance is 0, skipping division" << std::endl;
  }

  return f_divs;
}

float getLatentRatio(GANNetwork &fullNetwork, const Eigen::VectorXf &digitInput, std::mt19937 &rng)
{
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  Eigen::VectorXf z1(100), z2(100);
  for (int i = 0; i < 100; i++)
  {
    z1(i) = dist(rng);
    z2(i) = dist(rng);
  }

  Eigen::VectorXf input1(110);
  input1 << z1, digitInput;
  Eigen::VectorXf input2(110);
  input2 << z2, digitInput;

  forwardPass(fullNetwork.generator, input1);
  Eigen::VectorXf x1 = fullNetwork.generator.layers.back().a;
  forwardPass(fullNetwork.generator, input2);
  Eigen::VectorXf x2 = fullNetwork.generator.layers.back().a;

  float latentDist = (z1 - z2).norm();
  latentDist = std::max(latentDist, 1e-6f);

  float outputDist = (x1 - x2).norm() / std::sqrt(x1.size());

  float ratio = outputDist / latentDist;
  return ratio;
}