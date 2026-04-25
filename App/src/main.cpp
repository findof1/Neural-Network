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

    AppInterface ui;
    ui.show();

    return app.exec();
}