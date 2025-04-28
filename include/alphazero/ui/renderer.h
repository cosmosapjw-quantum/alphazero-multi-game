// include/alphazero/ui/renderer.h
#ifndef RENDERER_H
#define RENDERER_H

#include <string>
#include <memory>
#include <functional>

namespace alphazero {
namespace ui {

/**
 * @brief Abstract renderer interface for game visualization
 * 
 * This class defines the interface for rendering game boards and pieces.
 * Different implementations can be used for different output methods (text, GUI, etc.).
 */
class Renderer {
public:
    virtual ~Renderer() = default;
    
    /**
     * @brief Clear the rendering surface
     */
    virtual void clear() = 0;
    
    /**
     * @brief Draw a line between two points
     * 
     * @param x1 Start X coordinate
     * @param y1 Start Y coordinate
     * @param x2 End X coordinate
     * @param y2 End Y coordinate
     * @param color Color as string (format depends on implementation)
     * @param thickness Line thickness
     */
    virtual void drawLine(int x1, int y1, int x2, int y2, 
                         const std::string& color = "black", int thickness = 1) = 0;
    
    /**
     * @brief Draw a rectangle
     * 
     * @param x X coordinate of top-left corner
     * @param y Y coordinate of top-left corner
     * @param width Rectangle width
     * @param height Rectangle height
     * @param fillColor Fill color (empty for no fill)
     * @param outlineColor Outline color (empty for no outline)
     * @param thickness Outline thickness
     */
    virtual void drawRect(int x, int y, int width, int height,
                         const std::string& fillColor = "",
                         const std::string& outlineColor = "black",
                         int thickness = 1) = 0;
    
    /**
     * @brief Draw a circle
     * 
     * @param x X coordinate of center
     * @param y Y coordinate of center
     * @param radius Circle radius
     * @param fillColor Fill color (empty for no fill)
     * @param outlineColor Outline color (empty for no outline)
     * @param thickness Outline thickness
     */
    virtual void drawCircle(int x, int y, int radius,
                           const std::string& fillColor = "",
                           const std::string& outlineColor = "black",
                           int thickness = 1) = 0;
    
    /**
     * @brief Draw text
     * 
     * @param x X coordinate
     * @param y Y coordinate
     * @param text Text to draw
     * @param fontSize Font size
     * @param color Text color
     * @param centered Whether to center the text at the coordinates
     */
    virtual void drawText(int x, int y, const std::string& text, 
                         int fontSize = 12,
                         const std::string& color = "black",
                         bool centered = false) = 0;
    
    /**
     * @brief Draw an image
     * 
     * @param x X coordinate
     * @param y Y coordinate
     * @param width Image width
     * @param height Image height
     * @param imagePath Path to image file
     */
    virtual void drawImage(int x, int y, int width, int height,
                          const std::string& imagePath) = 0;
    
    /**
     * @brief Render to the display or output
     */
    virtual void render() = 0;
    
    /**
     * @brief Set the viewport size
     * 
     * @param width Viewport width
     * @param height Viewport height
     */
    virtual void setSize(int width, int height) = 0;
    
    /**
     * @brief Get the viewport width
     * 
     * @return Viewport width
     */
    virtual int getWidth() const = 0;
    
    /**
     * @brief Get the viewport height
     * 
     * @return Viewport height
     */
    virtual int getHeight() const = 0;
    
    /**
     * @brief Create a text renderer
     * 
     * @return Shared pointer to a text-based renderer
     */
    static std::shared_ptr<Renderer> createTextRenderer();
};

/**
 * @brief Implementation of Renderer that outputs ASCII/Unicode text
 */
class TextRenderer : public Renderer {
public:
    /**
     * @brief Constructor
     * 
     * @param width Width of the text grid
     * @param height Height of the text grid
     */
    TextRenderer(int width = 80, int height = 25);
    
    void clear() override;
    
    void drawLine(int x1, int y1, int x2, int y2, 
                 const std::string& color = "black", int thickness = 1) override;
    
    void drawRect(int x, int y, int width, int height,
                 const std::string& fillColor = "",
                 const std::string& outlineColor = "black",
                 int thickness = 1) override;
    
    void drawCircle(int x, int y, int radius,
                   const std::string& fillColor = "",
                   const std::string& outlineColor = "black",
                   int thickness = 1) override;
    
    void drawText(int x, int y, const std::string& text, 
                 int fontSize = 12,
                 const std::string& color = "black",
                 bool centered = false) override;
    
    void drawImage(int x, int y, int width, int height,
                  const std::string& imagePath) override;
    
    void render() override;
    
    void setSize(int width, int height) override;
    
    int getWidth() const override { return width_; }
    
    int getHeight() const override { return height_; }
    
    /**
     * @brief Get the rendered text
     * 
     * @return String containing the rendered text
     */
    std::string getText() const;
    
    /**
     * @brief Set output callback
     * 
     * @param callback Function to call with the rendered text
     */
    void setOutputCallback(std::function<void(const std::string&)> callback);
    
private:
    int width_;
    int height_;
    std::vector<std::vector<char>> grid_;
    std::function<void(const std::string&)> outputCallback_;
};

} // namespace ui
} // namespace alphazero

#endif // RENDERER_H