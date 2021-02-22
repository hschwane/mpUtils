/*
 * mpUtils
 * Sprite2DTools.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include <exception>
#include "mpUtils/Graphics/ContentCreation/Sprite2DTools.h"
#include "mpUtils/Misc/Image.h"
#include "mpUtils/Graphics/Gui/ImGui.h"
#include "mpUtils/Graphics/Gui/ImGuiWindows.h"
#include "mpUtils/external/tinyfd/tinyfiledialogs.h"
#include "mpUtils/Misc/additionalMath.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

Sprite2DData makeSimpleSprite(std::string pathToImage, std::string workDir)
{
    // load image
    auto img = Image8(workDir+pathToImage);

    // check for transparency
    bool hasSemiTransparancy=false;
    for(int i=0; i < img.numPixels(); ++i) {
        if( img(i)[3] > 0 && img(i)[3] < 255)
            hasSemiTransparancy = true;
    }

    // fill data
    Sprite2DData data;
    data.tileFactor = 1.0f;
    data.forward = 0.0f;
    data.texture = pathToImage;
    data.pivot = {0,0};
    data.rectInImage = {0,0,img.width(),img.height()};
    data.worldSize = {1.0f, float(img.height()) / float(img.width())};
    data.semiTransparent = hasSemiTransparancy;
    data.spritesheet = "";
    return data;
}

void SpriteEditor::show(bool* show, bool drawAsChild)
{
    std::string title = std::string("Sprite Editor - ") + (m_hasUnsavedChanges ? "*" : "") + m_filename + m_id;

    // draw actual window
    bool visible;
    if(drawAsChild) {
        visible = ImGui::BeginChild(title.c_str());
    } else {
        ImGui::SetNextWindowSize(ImVec2(480,300),ImGuiCond_FirstUseEver);
        visible = ImGui::Begin(title.c_str(), show);
    }

    if(visible) {

        // split for image preview and controles
        float widthLeft = std::fmin(ImGui::GetWindowContentRegionWidth() * 0.5f, 255.0f);
        float widthRight = ImGui::GetWindowContentRegionWidth() - widthLeft -ImGui::GetStyle().ItemSpacing.x;
        float height = ImGui::GetContentRegionAvail().y-60.0f;

        if(ImGui::Button(ICON_FA_FILE_IMAGE_O))
            selectTextureWithFileDlg();
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("Select file");

        ImGui::SameLine();
        if(ImGui::InputText("Image file", &m_data.texture, ImGuiInputTextFlags_EnterReturnsTrue)
           || ImGui::IsItemDeactivatedAfterEdit()) {
            m_hasUnsavedChanges = true;
            tryLoadTexture();
        }
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", m_data.texture.c_str());

        ImGui::SameLine();
        if(ImGui::Button("Auto Fill"))
            autoFillAll();
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("Set sensible default values based on the loaded image.");

        ImGui::Separator();

        // draw left side with control elements
        ImGui::BeginChild("left", ImVec2(widthLeft, height));
        {
            m_hasUnsavedChanges |= ImGui::InputText("Name", &m_filename);
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("%s",m_filename.c_str());

            m_hasUnsavedChanges |= ImGui::Checkbox("Semitransparent", &m_data.semiTransparent);
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("Should be checked if the image contains pixels with alpha values not 0 or 255.");

            ImGui::SameLine();
            if(ImGui::Button(ICON_FA_MAGIC))
                autoDetectTransparancy();
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("Auto detect transparency in cropped subregion.");

            m_hasUnsavedChanges |= ImGui::DragInt4("crop", glm::value_ptr(m_data.rectInImage));
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("Crop a rectangle from the original image.");
            if(m_image && ImGui::IsItemDeactivatedAfterEdit()) {
                m_data.rectInImage.x = std::min(std::max(m_data.rectInImage.x,0),m_image->width());
                m_data.rectInImage.y = std::min(std::max(m_data.rectInImage.y,0),m_image->height());
                m_data.rectInImage.z = std::min(std::max(m_data.rectInImage.z,0),m_image->width());
                m_data.rectInImage.w = std::min(std::max(m_data.rectInImage.w,0),m_image->height());

                m_data.rectInImage.z = m_data.rectInImage.z > m_data.rectInImage.x ? m_data.rectInImage.z : m_data.rectInImage.x+1;
                m_data.rectInImage.w = m_data.rectInImage.w > m_data.rectInImage.y ? m_data.rectInImage.w : m_data.rectInImage.y+1;
            }

            ImGui::SameLine();
            if(ImGui::Button(ICON_FA_ARROWS_ALT))
                setCropToFullImage();
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("Set crop size to cover the full image.");

            m_hasUnsavedChanges |= ImGui::DragFloat2("World size", glm::value_ptr(m_data.worldSize));
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("Size of the sprite in game world coordinates.");

            float forwardDeg = deg(m_data.forward);
            if(ImGui::DragFloat("Foraward", &forwardDeg)) {
                m_data.forward = rad(std::fmod(forwardDeg,360.0f));
                m_hasUnsavedChanges = true;
            }
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("Direction considered forward or up in degrees from the top. (rotates sprite)");

            m_hasUnsavedChanges |= ImGui::DragFloat2("Pivot", glm::value_ptr(m_data.pivot), 0.01);
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("Point around which the sprite will be moved and rotated.\n(0,0) is the center, (1,1) the upper right corner.");

            m_hasUnsavedChanges |= ImGui::DragFloat("Tile factor", &m_data.tileFactor, 0.01);
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("Amount of times the texture is repeated horizontally and vertically inside the sprite.");
        }
        ImGui::EndChild();

        // calculate size of image preview
        ImGui::SameLine();
        // draw right side with image preview
        ImGui::BeginChild("right", ImVec2(widthRight, height),false,ImGuiWindowFlags_NoScrollbar);
        {
            // prepare drawing a border
            glm::vec2 imageBorderSize;
            if(m_image) {
                float aspect = float(m_image->height()) / float(m_image->width());
                imageBorderSize.x = std::min(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / aspect);
                imageBorderSize.y = std::min(ImGui::GetContentRegionAvail().x * aspect, ImGui::GetContentRegionAvail().y);
            } else
                imageBorderSize = glm::vec2(std::min(ImGui::GetContentRegionAvail().x,ImGui::GetContentRegionAvail().y));
            glm::vec2 previewSize = imageBorderSize - 2*ImGui::GetStyle().FramePadding.x;

            glm::vec2 p0 = ImGui::GetCursorScreenPos(); // p0 for the border
            ImGui::SetCursorScreenPos(p0 + ImGui::GetStyle().FramePadding.x);
            if(m_texture && m_image) {
                glm::vec2 previewStartPos = ImGui::GetCursorScreenPos();
                ImGui::Image((void*)(intptr_t)static_cast<GLuint>(*m_texture), previewSize, ImVec2(0, 1), ImVec2(1, 0));
                drawImageOverlay(previewStartPos,previewSize);
            } else {
                ImGui::BeginHorizontal("hor", previewSize);
                ImGui::Spring(0.5f);
                ICON_BEGIN();
                ImGui::Text(ICON_FA_PICTURE_O);
                ICON_END();
                ImGui::Spring(0.5f);
                ImGui::EndHorizontal();
                if(ImGui::IsItemHovered())
                    ImGui::SetTooltip("Select file");
                if(ImGui::IsItemClicked(0)) {
                    selectTextureWithFileDlg();
                }
            }
            // draw border
            ImGui::GetWindowDrawList()->AddRect(p0, p0+imageBorderSize, ImColor(ImGui::GetStyleColorVec4(ImGuiCol_Border)));
        }
        ImGui::EndChild();

        ImGui::Button(ICON_FA_FILE_O" New");
        ImGui::SameLine();
        ImGui::Button(ICON_FA_UNDO" Reset");
        ImGui::SameLine();
        ImGui::Button(ICON_FA_FLOPPY_O" Save");
        ImGui::SameLine();
        if(!drawAsChild)
            ImGui::Button(ICON_FA_BAN" Cancel");
    }

    if(drawAsChild)
        ImGui::EndChild();
    else
        ImGui::End();
}

void SpriteEditor::tryLoadTexture()
{
    try {
        m_image = std::make_unique<Image8>(m_workDir+m_data.texture);
        m_texture = makeTextureFromImage(*m_image);
    } catch(const std::exception& e) {
        logERROR("SpriteEditor") << "Could not load image " << m_workDir+m_data.texture << ". Error: " << e.what();
        m_image = nullptr;
        m_texture = nullptr;
        ImGui::SimpleModal("Error", std::string("Error opening image: ") + e.what(), {"ok"}, ICON_FA_TIMES_CIRCLE);
    }
}

void SpriteEditor::selectTextureWithFileDlg()
{
    char const* filterPatterns[] = {"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tga", "*.psd"};
    const char* file = tinyfd_openFileDialog("Select image file", m_data.texture.c_str(), 6, filterPatterns,
                                             "Image Files", false);
    if(file) {
        m_data.texture = file;
        m_hasUnsavedChanges = true;
        tryLoadTexture();
    }
}

void SpriteEditor::autoDetectTransparancy()
{
    if(m_image) {
        m_data.semiTransparent = false;
        for(int y=m_data.rectInImage.y; y<m_data.rectInImage.w; ++y)
            for(int x=m_data.rectInImage.x; x<m_data.rectInImage.z; ++x)
                if(m_image->gCoord(x,y)[3] > 0 && m_image->gCoord(x,y)[3] < 255)
                    m_data.semiTransparent = true;
        m_hasUnsavedChanges = true;
    }
}

void SpriteEditor::setCropToFullImage()
{
    if(m_image) {
        m_data.rectInImage = {0,0,m_image->width(), m_image->height()};
        m_hasUnsavedChanges = true;
    }
}

void SpriteEditor::autoFillAll()
{
    if(m_image) {
        setCropToFullImage();
        autoDetectTransparancy();

        auto a = m_data.texture.find_last_of('/')+1;
        auto b = m_data.texture.find_last_of('.');

        m_filename = m_data.texture.substr(a,b-a);

        m_data.tileFactor = 1.0f;
        m_data.forward = 0.0f;
        m_data.pivot = {0, 0};
        m_data.worldSize = {1.0f, float(m_image->height()) / float(m_image->width())};
        m_data.spritesheet = "";
    }
}

void SpriteEditor::drawImageOverlay(const glm::vec2& previewStartPos, const glm::vec2& previewSize)
{
    auto dl = ImGui::GetWindowDrawList();

    // crop rectangle
    auto color = ImColor(ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
    glm::vec2 imgOrigInScreenCS = previewStartPos;// lower left of the image in screen coordinates
    imgOrigInScreenCS.y += previewSize.y;
    glm::vec2 imageSize(m_image->width(), m_image->height());
    glm::vec2 sizeFactor = previewSize / imageSize;

    glm::vec2 bl = imgOrigInScreenCS + glm::vec2(m_data.rectInImage.x, -m_data.rectInImage.y) * sizeFactor;
    glm::vec2 tr = imgOrigInScreenCS + glm::vec2(m_data.rectInImage.z, -m_data.rectInImage.w) * sizeFactor;
    glm::vec2 tl = {bl.x, tr.y};
    glm::vec2 br = {tr.x, bl.y};
    glm::vec2 b = {bl.x + (tr.x - bl.x) * 0.5f, bl.y};
    glm::vec2 l = {bl.x, bl.y + (tr.y - bl.y) * 0.5f};
    glm::vec2 t = {bl.x + (tr.x - bl.x) * 0.5f, tr.y};
    glm::vec2 r = {tr.x, bl.y + (tr.y - bl.y) * 0.5f};
    glm::vec2 center = {bl.x + (tr.x - bl.x) * 0.5f, bl.y + (tr.y - bl.y) * 0.5f};

    // draw darkened background
    dl->AddRectFilled(previewStartPos, glm::vec2(previewStartPos.x+previewSize.x, tr.y), ImColor(ImVec4(0.3f,0.3f,0.3f,0.5f)) );
    dl->AddRectFilled(glm::vec2(previewStartPos.x,tr.y), bl, ImColor(ImVec4(0.3f,0.3f,0.3f,0.5f)) );
    dl->AddRectFilled(tr, glm::vec2(previewStartPos.x+previewSize.x,br.y), ImColor(ImVec4(0.3f,0.3f,0.3f,0.5f)) );
    dl->AddRectFilled( glm::vec2(previewStartPos.x,bl.y), previewStartPos+previewSize, ImColor(ImVec4(0.3f,0.3f,0.3f,0.5f)) );

    // draw actual crop rectangle
    dl->AddRect(tl, br, color);

    // draw crop rect handles
    glm::vec2 handleRad(4.0f, 4.0f); // handle "radius"

    dl->AddRect(bl - handleRad, bl + handleRad, color);
    ImGui::SetCursorScreenPos(bl - handleRad);
    ImGui::InvisibleButton("bl-handle", handleRad * 2);
    if(ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNESW);
    if(ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
        m_data.rectInImage.x = (ImGui::GetIO().MousePos.x - imgOrigInScreenCS.x)  / sizeFactor.x;
        m_data.rectInImage.y = (imgOrigInScreenCS.y - ImGui::GetIO().MousePos.y)  / sizeFactor.y;
    }

    dl->AddRect(tr - handleRad, tr + handleRad, color);
    ImGui::SetCursorScreenPos(tr - handleRad);
    ImGui::InvisibleButton("tr-handle", handleRad * 2);
    if(ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNESW);
    if(ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
        m_data.rectInImage.z = (ImGui::GetIO().MousePos.x - imgOrigInScreenCS.x)  / sizeFactor.x;
        m_data.rectInImage.w = (imgOrigInScreenCS.y - ImGui::GetIO().MousePos.y)  / sizeFactor.y;
    }

    dl->AddRect(tl - handleRad, tl + handleRad, color);
    ImGui::SetCursorScreenPos(tl - handleRad);
    ImGui::InvisibleButton("tl-handle", handleRad * 2);
    if(ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNWSE);
    if(ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
        m_data.rectInImage.x = (ImGui::GetIO().MousePos.x - imgOrigInScreenCS.x)  / sizeFactor.x;
        m_data.rectInImage.w = (imgOrigInScreenCS.y - ImGui::GetIO().MousePos.y)  / sizeFactor.y;
    }

    dl->AddRect(br - handleRad, br + handleRad, color);
    ImGui::SetCursorScreenPos(br - handleRad);
    ImGui::InvisibleButton("br-handle", handleRad * 2);
    if(ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNWSE);
    if(ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
        m_data.rectInImage.z = (ImGui::GetIO().MousePos.x - imgOrigInScreenCS.x)  / sizeFactor.x;
        m_data.rectInImage.y = (imgOrigInScreenCS.y -ImGui::GetIO().MousePos.y)  / sizeFactor.y;
    }

    dl->AddRect(b - handleRad, b + handleRad, color);
    ImGui::SetCursorScreenPos(b - handleRad);
    ImGui::InvisibleButton("b-handle", handleRad * 2);
    if(ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    if(ImGui::IsItemActive() && ImGui::IsMouseDragging(0))
        m_data.rectInImage.y = (imgOrigInScreenCS.y-ImGui::GetIO().MousePos.y)  / sizeFactor.y;


    dl->AddRect(l - handleRad, l + handleRad, color);
    ImGui::SetCursorScreenPos(l - handleRad);
    ImGui::InvisibleButton("l-handle", handleRad * 2);
    if(ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    if(ImGui::IsItemActive() && ImGui::IsMouseDragging(0))
        m_data.rectInImage.x = (ImGui::GetIO().MousePos.x - imgOrigInScreenCS.x)  / sizeFactor.x;

    dl->AddRect(t - handleRad, t + handleRad, color);
    ImGui::SetCursorScreenPos(t - handleRad);
    ImGui::InvisibleButton("t-handle", handleRad * 2);
    if(ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    if(ImGui::IsItemActive() && ImGui::IsMouseDragging(0))
        m_data.rectInImage.w = (imgOrigInScreenCS.y-ImGui::GetIO().MousePos.y)  / sizeFactor.y;

    dl->AddRect(r - handleRad, r + handleRad, color);
    ImGui::SetCursorScreenPos(r - handleRad);
    ImGui::InvisibleButton("r-handle", handleRad * 2);
    if(ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    if(ImGui::IsItemActive() && ImGui::IsMouseDragging(0))
        m_data.rectInImage.z = (ImGui::GetIO().MousePos.x - imgOrigInScreenCS.x)  / sizeFactor.x;

    // crosshair in the middle
    dl->AddLine( center-glm::vec2(0,handleRad.y*2), center+glm::vec2(0,handleRad.y*2), color);
    dl->AddLine( center-glm::vec2(handleRad.x*2,0), center+glm::vec2(handleRad.x*2,0), color);

    // direction indicator
    glm::vec2 p1(-4,-3);
    glm::vec2 p2(0,-25);
    glm::vec2 p3(4,-3);
    dl->AddTriangleFilled( center+glm::rotate(p1,m_data.forward),center+glm::rotate(p2,m_data.forward),
                           center+glm::rotate(p3,m_data.forward), color);

    // middle button
    ImGui::SetCursorScreenPos(tl + glm::vec2(2.0f));
    ImGui::InvisibleButton("overallHandle", br -tl -glm::vec2(4.0f));
    static glm::vec4 cropOffsets;
    if(ImGui::IsItemClicked(0)) {
        cropOffsets.x = m_data.rectInImage.x - (ImGui::GetIO().MousePos.x - imgOrigInScreenCS.x)  / sizeFactor.x;
        cropOffsets.y = m_data.rectInImage.y - (imgOrigInScreenCS.y - ImGui::GetIO().MousePos.y)  / sizeFactor.y;
        cropOffsets.z = m_data.rectInImage.z - (ImGui::GetIO().MousePos.x - imgOrigInScreenCS.x)  / sizeFactor.x;
        cropOffsets.w = m_data.rectInImage.w - (imgOrigInScreenCS.y - ImGui::GetIO().MousePos.y)  / sizeFactor.y;
    }
    if(ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
        int px = int(cropOffsets.x + (ImGui::GetIO().MousePos.x - imgOrigInScreenCS.x)  / sizeFactor.x);
        int py = int(cropOffsets.y + (imgOrigInScreenCS.y - ImGui::GetIO().MousePos.y)  / sizeFactor.y);
        int pz = int(cropOffsets.z + (ImGui::GetIO().MousePos.x - imgOrigInScreenCS.x)  / sizeFactor.x);
        int pw = int(cropOffsets.w + (imgOrigInScreenCS.y - ImGui::GetIO().MousePos.y)  / sizeFactor.y);

        if(px < 0) {
            m_data.rectInImage.x = 0;
            m_data.rectInImage.z = pz-px;
        } else if(pz > m_image->width()) {
            m_data.rectInImage.x = m_image->width() - (pz-px);
            m_data.rectInImage.z = m_image->width();
        } else {
            m_data.rectInImage.x = px;
            m_data.rectInImage.z = pz;
        }

        if(py < 0) {
            m_data.rectInImage.y = 0;
            m_data.rectInImage.w = pw-py;
        } else if(pw > m_image->height()) {
            m_data.rectInImage.y = m_image->height() - (pw-py);
            m_data.rectInImage.w = m_image->height();
        } else {
            m_data.rectInImage.y = py;
            m_data.rectInImage.w = pw;
        }
    }
    ImGui::SetItemAllowOverlap();

    // pivot
    glm::vec2 cropRegionSize = br - tl;
    glm::vec2 worldSizeFactor = cropRegionSize / m_data.worldSize * 0.5f;
    glm::vec2 pivotPos = {center.x+m_data.pivot.x*worldSizeFactor.x,center.y-m_data.pivot.y*worldSizeFactor.y};

    dl->AddLine( pivotPos-glm::vec2(0,handleRad.y*2), pivotPos+glm::vec2(0,handleRad.y*2), color);
    dl->AddLine( pivotPos-glm::vec2(handleRad.x*2,0), pivotPos+glm::vec2(handleRad.x*2,0), color);
    dl->AddCircle(pivotPos,handleRad.x,color);

    ImGui::SetCursorScreenPos(pivotPos - handleRad);
    ImGui::InvisibleButton("pivot-handle", handleRad * 2);
    if(ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
    if(ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
        m_data.pivot.x = (ImGui::GetIO().MousePos.x - center.x) / worldSizeFactor.x;
        m_data.pivot.y = (center.y-ImGui::GetIO().MousePos.y) / worldSizeFactor.y;
    }
}

}}