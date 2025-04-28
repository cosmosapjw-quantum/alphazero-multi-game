// src/ui/renderer.cpp
#include "alphazero/ui/renderer.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace alphazero {
namespace ui {

// Factory method to create a text renderer
std::shared_ptr<Renderer> Renderer::createTextRenderer() {
    return std::make_shared<TextRenderer>();
}

// TextRenderer implementation
TextRenderer::TextRenderer(int width, int height)
    : width_(width), height_(height) {
    
    // Initialize grid with spaces
    grid_.resize(height_, std::vector<char>(width_, ' '));
    
    // Default output callback
    outputCallback_ = [](const std::string& text) {
        std::cout << text << std::endl;
    };
}

void TextRenderer::clear() {
    // Reset grid with spaces
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            grid_[y][x] = ' ';
        }
    }
}

void TextRenderer::drawLine(int x1, int y1, int x2, int y2,
                           const std::string& color, int thickness) {
    // Skip color and thickness in text mode
    
    // Bresenham's line algorithm
    int dx = std::abs(x2 - x1);
    int dy = std::abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;
    
    while (true) {
        // Draw point if within bounds
        if (x1 >= 0 && x1 < width_ && y1 >= 0 && y1 < height_) {
            grid_[y1][x1] = '+';
        }
        
        if (x1 == x2 && y1 == y2) {
            break;
        }
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }
}

void TextRenderer::drawRect(int x, int y, int width, int height,
                          const std::string& fillColor,
                          const std::string& outlineColor,
                          int thickness) {
    // Skip color and thickness in text mode
    
    // Draw outline
    if (!outlineColor.empty()) {
        drawLine(x, y, x + width - 1, y);                   // Top
        drawLine(x, y + height - 1, x + width - 1, y + height - 1);  // Bottom
        drawLine(x, y, x, y + height - 1);                  // Left
        drawLine(x + width - 1, y, x + width - 1, y + height - 1);   // Right
    }
    
    // Fill interior
    if (!fillColor.empty()) {
        for (int py = y + 1; py < y + height - 1; ++py) {
            for (int px = x + 1; px < x + width - 1; ++px) {
                if (px >= 0 && px < width_ && py >= 0 && py < height_) {
                    grid_[py][px] = '#';
                }
            }
        }
    }
}

void TextRenderer::drawCircle(int x, int y, int radius,
                            const std::string& fillColor,
                            const std::string& outlineColor,
                            int thickness) {
    // Skip color and thickness in text mode
    
    // Midpoint circle algorithm
    int px = radius;
    int py = 0;
    int err = 0;
    
    while (px >= py) {
        // Draw 8 octants
        auto drawPixel = [&](int px, int py) {
            if (px >= 0 && px < width_ && py >= 0 && py < height_) {
                grid_[py][px] = 'O';
            }
        };
        
        drawPixel(x + px, y + py);
        drawPixel(x + py, y + px);
        drawPixel(x - py, y + px);
        drawPixel(x - px, y + py);
        drawPixel(x - px, y - py);
        drawPixel(x - py, y - px);
        drawPixel(x + py, y - px);
        drawPixel(x + px, y - py);
        
        py++;
        if (err <= 0) {
            err += 2 * py + 1;
        }
        if (err > 0) {
            px--;
            err -= 2 * px + 1;
        }
    }
    
    // Fill interior (simplified)
    if (!fillColor.empty()) {
        for (int py = y - radius + 1; py <= y + radius - 1; ++py) {
            for (int px = x - radius + 1; px <= x + radius - 1; ++px) {
                int dx = px - x;
                int dy = py - y;
                if (dx * dx + dy * dy < (radius - 1) * (radius - 1)) {
                    if (px >= 0 && px < width_ && py >= 0 && py < height_) {
                        grid_[py][px] = '*';
                    }
                }
            }
        }
    }
}

void TextRenderer::drawText(int x, int y, const std::string& text,
                          int fontSize, const std::string& color,
                          bool centered) {
    // Skip font size and color in text mode
    
    // Calculate start position
    int startX = x;
    if (centered) {
        startX = x - static_cast<int>(text.length()) / 2;
    }
    
    // Draw text
    for (size_t i = 0; i < text.length(); ++i) {
        int px = startX + static_cast<int>(i);
        if (px >= 0 && px < width_ && y >= 0 && y < height_) {
            grid_[y][px] = text[i];
        }
    }
}

void TextRenderer::drawImage(int x, int y, int width, int height,
                           const std::string& imagePath) {
    // Not applicable in text mode - draw a placeholder
    drawRect(x, y, width, height, "", "I", 1);
    
    // Draw text in center
    std::string placeholder = "IMG";
    drawText(x + width / 2, y + height / 2, placeholder, 0, "", true);
}

void TextRenderer::render() {
    // Build string representation
    std::ostringstream ss;
    
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            ss << grid_[y][x];
        }
        ss << '\n';
    }
    
    // Call output callback
    if (outputCallback_) {
        outputCallback_(ss.str());
    }
}

void TextRenderer::setSize(int width, int height) {
    width_ = width;
    height_ = height;
    
    // Resize grid
    grid_.resize(height_);
    for (auto& row : grid_) {
        row.resize(width_, ' ');
    }
}

std::string TextRenderer::getText() const {
    std::ostringstream ss;
    
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            ss << grid_[y][x];
        }
        ss << '\n';
    }
    
    return ss.str();
}

void TextRenderer::setOutputCallback(std::function<void(const std::string&)> callback) {
    if (callback) {
        outputCallback_ = callback;
    }
}

} // namespace ui
} // namespace alphazero