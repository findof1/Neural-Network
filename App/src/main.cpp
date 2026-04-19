#include <QApplication>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Dense>
#include <random>
#include "styleGlobals.hpp"
#include "styleManager.hpp"
#include "interface.hpp"
#include "ganUtils.hpp"

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    gStyleManager.init(&app);
    gStyleManager.setDarkStyle();

    // just doing some testing
    /*
    GANNetwork net;
    net.generator.config.epochs = 5;
    net.generator.config.epochDisplayInterval = 1;
    net.generator.config.batchSize = 32;
    addLayer(net.generator, 100, 128, ReLU);
    addLayer(net.generator, 128, 256, ReLU);
    addLayer(net.generator, 256, 512, ReLU);
    addLayer(net.generator, 512, 784, Sigmoid);

    addLayer(net.discriminator, 784, 512, ReLU);
    addLayer(net.discriminator, 512, 256, ReLU);
    addLayer(net.discriminator, 256, 128, ReLU);
    addLayer(net.discriminator, 128, 2, Sigmoid);

    QString base = QCoreApplication::applicationDirPath();
    Dataset dataset = loadMNISTCSV((base + "/mnist_train/mnist_train.csv").toStdString());

    std::mt19937 rng(std::random_device{}());
    std::cout << "Beginning Training" << std::endl;
    for (int epoch = 0; epoch < net.generator.config.epochs; epoch++)
    {
        std::shuffle(dataset.samples.begin(), dataset.samples.end(), rng);

        float totalLossGen = 0.0f;
        float totalLossDis = 0.0f;
        int count = 0;

        for (auto &sample : dataset.samples)
        {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

            const int latentSize = 100;

            Eigen::VectorXf z(latentSize);

            for (int i = 0; i < latentSize; i++)
                z(i) = dist(rng);

            forwardPass(net.generator, z);
            Eigen::VectorXf fakeData = net.generator.layers.back().a;
            forwardPass(net.discriminator, fakeData);
            Eigen::VectorXf fakeOut = net.discriminator.layers.back().a;

            Sample fakeSample;
            fakeSample.inputs = fakeData;
            fakeSample.targets = Eigen::VectorXf(2);
            fakeSample.targets << 0.0f, 1.0f;
            discriminatorBackpropagation(net, fakeSample);

            Eigen::VectorXf fakeGradient = net.discriminator.layers[0].W.transpose() * net.discriminator.layers[0].delta;
            generatorBackpropagation(net, z, fakeGradient);

            Sample realCopy = sample;
            realCopy.targets = Eigen::VectorXf(2);
            realCopy.targets << 1.0f, 0.0f;

            forwardPass(net.discriminator, realCopy.inputs);
            Eigen::VectorXf realOut = net.discriminator.layers.back().a;
            discriminatorBackpropagation(net, realCopy);

            totalLossGen += computeGeneratorLoss(fakeOut);
            totalLossDis += computeDiscriminatorLoss(realOut, fakeOut);
            count++;

            if (count % net.generator.config.batchSize == 0)
            {
                applyGradients(net.generator);
                applyGradients(net.discriminator);
            }

            if (count % 6000 == 0)
            {
                std::cout << "6000 samples complete" << std::endl;
            }
        }

        if ((epoch + 1) % net.generator.config.epochDisplayInterval == 0)
        {
            float lossGen = totalLossGen / dataset.samples.size();
            float lossDis = totalLossDis / dataset.samples.size();
            std::cout << "Epoch: " << epoch + 1 << std::endl;
            std::cout << "Generator Loss: " << lossGen << std::endl;
            std::cout << "Discriminator Loss: " << lossDis << std::endl;
        }
    }

    std::cin.get();*/

    AppInterface ui;
    ui.show();

    return app.exec();
}