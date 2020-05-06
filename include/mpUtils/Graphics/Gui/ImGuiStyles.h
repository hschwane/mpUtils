/*
 * mpUtils
 * ImGuiStyles.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_IMGUISTYLES_H
#define MPUTILS_IMGUISTYLES_H

// namespace
//--------------------
namespace ImGui { // we extend the imgui namespace inside mpu
//--------------------

/**
 * @brief default imgui dark theme
 */
void StyleImGuiDefaultDark();

/**
 * @brief default imgui light theme
 */
void StyleImGuiDefaultLight();

/**
 * @brief classic im gui colo scheme.
 */
void StyleImGuiDefaultClassic();

/**
 * @brief sets the ImGui-Style to "CorporateGreyFlat". Taken from https://github.com/ocornut/imgui/issues/707.
 */
void StyleCorporateGreyFlat();

/**
 * @brief sets the ImGui-Style to "CorporateGrey". Same Colors as the flat variant but more depth. Taken from https://github.com/ocornut/imgui/issues/707.
 */
void StyleCorporateGrey();

/**
 * @brief simulation of itelliJ Darcula theme. Taken from https://github.com/ocornut/imgui/issues/707.
 */
void StyleDarcula();

/**
 * @brief colorfull light style. Taken from https://github.com/ocornut/imgui/issues/707.
 */
void StylePagghiu();

/**
 * @brief light green color scheme. Taken from https://github.com/ocornut/imgui/issues/707.
 */
void StyleLightGreen();

/**
 * @brief clean back and blue color scheme. Taken from https://github.com/ocornut/imgui/issues/707#issuecomment-512669512
 */
void StyleYetAnotherDarktheme();

/**
 * @brief dar base color with golden details. Taken from https://github.com/ocornut/imgui/issues/707#issuecomment-622934113
 */
void StyleGoldAndBlack();

}

#endif //MPUTILS_IMGUISTYLES_H
