#pragma once
#include <QApplication>
#include <QPalette>

struct StyleData
{
  QColor window;
  QColor base;
  QColor altBase;

  QColor text;
  QColor textDisabled;

  QColor windowText;
  QColor windowTextDisabled;

  QColor buttonText;
  QColor buttonTextDisabled;

  QColor placeholderText;
  QColor placeholderTextDisabled;

  QColor button;
  QColor buttonDisabled;

  QColor hightlight;
  QColor hightlightDisabled;

  QColor hightlightedText;
  QColor hightlightedTextDisabled;

  QColor toolTipBase;
  QColor toolTipText;
  QColor toolTipTextDisabled;

  QColor brightText;
  QColor accent;
};

class StyleManager
{
public:
  StyleManager()
  {
  }

  void init(QApplication *appRef)
  {
    app = appRef;
  }

  StyleData getDarkStyleData()
  {
    StyleData s;

    s.window = QColor("#1e1e1e");
    s.base = QColor("#2a2a2a");
    s.altBase = QColor("#242424");

    s.text = QColor("#ffffff");
    s.textDisabled = QColor("#777777");

    s.windowText = QColor("#ffffff");
    s.windowTextDisabled = QColor("#777777");

    s.buttonText = QColor("#ffffff");
    s.buttonTextDisabled = QColor("#777777");

    s.placeholderText = QColor("#888888");
    s.placeholderTextDisabled = QColor("#555555");

    s.button = QColor("#2d2d2d");
    s.buttonDisabled = QColor("#2a2a2a");

    s.hightlight = QColor("#3a82f7");
    s.hightlightDisabled = QColor("#444444");

    s.hightlightedText = QColor("#ffffff");
    s.hightlightedTextDisabled = QColor("#aaaaaa");

    s.toolTipBase = QColor("#2a2a2a");
    s.toolTipText = QColor("#ffffff");
    s.toolTipTextDisabled = QColor("#777777");

    s.brightText = QColor("#ff5555");
    s.accent = QColor("#3a82f7");

    return s;
  }

  StyleData getLightStyleData()
  {
    StyleData s;

    s.window = QColor("#f0f0f0");
    s.base = QColor("#ffffff");
    s.altBase = QColor("#e8e8e8");

    s.text = QColor("#000000");
    s.textDisabled = QColor("#777777");

    s.windowText = QColor("#000000");
    s.windowTextDisabled = QColor("#777777");

    s.buttonText = QColor("#000000");
    s.buttonTextDisabled = QColor("#777777");

    s.placeholderText = QColor("#888888");
    s.placeholderTextDisabled = QColor("#aaaaaa");

    s.button = QColor("#e0e0e0");
    s.buttonDisabled = QColor("#d6d6d6");

    s.hightlight = QColor("#3a82f7");
    s.hightlightDisabled = QColor("#a0a0a0");

    s.hightlightedText = QColor("#ffffff");
    s.hightlightedTextDisabled = QColor("#f0f0f0");

    s.toolTipBase = QColor("#ffffff");
    s.toolTipText = QColor("#000000");
    s.toolTipTextDisabled = QColor("#777777");

    s.brightText = QColor("#ff0000");
    s.accent = QColor("#3a82f7");

    return s;
  }

  void applyStyle(const StyleData &s)
  {
    if (app == nullptr)
    {
      return;
    }

    currentStyle = s;
    app->setStyle("Fusion");

    QPalette palette;

    palette.setColor(QPalette::Active, QPalette::Window, s.window);
    palette.setColor(QPalette::Active, QPalette::Base, s.base);
    palette.setColor(QPalette::Active, QPalette::AlternateBase, s.altBase);

    palette.setColor(QPalette::Active, QPalette::Text, s.text);
    palette.setColor(QPalette::Active, QPalette::WindowText, s.windowText);
    palette.setColor(QPalette::Active, QPalette::ButtonText, s.buttonText);
    palette.setColor(QPalette::Active, QPalette::PlaceholderText, s.placeholderText);

    palette.setColor(QPalette::Active, QPalette::Button, s.button);

    palette.setColor(QPalette::Active, QPalette::Highlight, s.hightlight);
    palette.setColor(QPalette::Active, QPalette::HighlightedText, s.hightlightedText);

    palette.setColor(QPalette::Active, QPalette::ToolTipBase, s.toolTipBase);
    palette.setColor(QPalette::Active, QPalette::ToolTipText, s.toolTipText);

    palette.setColor(QPalette::Active, QPalette::BrightText, s.brightText);
    palette.setColor(QPalette::Active, QPalette::Accent, s.accent);

    palette.setColor(QPalette::Inactive, QPalette::Window, s.window);
    palette.setColor(QPalette::Inactive, QPalette::Base, s.base);
    palette.setColor(QPalette::Inactive, QPalette::AlternateBase, s.altBase);

    palette.setColor(QPalette::Inactive, QPalette::Text, s.text);
    palette.setColor(QPalette::Inactive, QPalette::WindowText, s.windowText);
    palette.setColor(QPalette::Inactive, QPalette::ButtonText, s.buttonText);
    palette.setColor(QPalette::Inactive, QPalette::PlaceholderText, s.placeholderText);

    palette.setColor(QPalette::Inactive, QPalette::Button, s.button);

    palette.setColor(QPalette::Inactive, QPalette::Highlight, s.hightlight);
    palette.setColor(QPalette::Inactive, QPalette::HighlightedText, s.hightlightedText);

    palette.setColor(QPalette::Inactive, QPalette::ToolTipBase, s.toolTipBase);
    palette.setColor(QPalette::Inactive, QPalette::ToolTipText, s.toolTipText);
    palette.setColor(QPalette::Inactive, QPalette::Accent, s.accent);

    palette.setColor(QPalette::Disabled, QPalette::Window, s.window);
    palette.setColor(QPalette::Disabled, QPalette::Base, s.base);
    palette.setColor(QPalette::Disabled, QPalette::AlternateBase, s.altBase);

    palette.setColor(QPalette::Disabled, QPalette::Text, s.textDisabled);
    palette.setColor(QPalette::Disabled, QPalette::WindowText, s.windowTextDisabled);
    palette.setColor(QPalette::Disabled, QPalette::ButtonText, s.buttonTextDisabled);
    palette.setColor(QPalette::Disabled, QPalette::PlaceholderText, s.placeholderTextDisabled);

    palette.setColor(QPalette::Disabled, QPalette::Button, s.buttonDisabled);

    palette.setColor(QPalette::Disabled, QPalette::Highlight, s.hightlightDisabled);
    palette.setColor(QPalette::Disabled, QPalette::HighlightedText, s.hightlightedTextDisabled);

    palette.setColor(QPalette::Disabled, QPalette::ToolTipBase, s.toolTipBase);
    palette.setColor(QPalette::Disabled, QPalette::ToolTipText, s.toolTipTextDisabled);

    app->setPalette(palette);
    app->setStyleSheet(R"()");
  }

  void setDarkStyle()
  {
    StyleData style = getDarkStyleData();
    applyStyle(style);
  }

  void setLightStyle()
  {
    StyleData style = getLightStyleData();
    applyStyle(style);
  }

  StyleData getCurrentStyle()
  {
    return currentStyle;
  }

private:
  StyleData currentStyle;
  QApplication *app;
};