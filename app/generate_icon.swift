import AppKit
import CoreGraphics
import Foundation

let size = CGSize(width: 1024, height: 1024)
let rep = NSBitmapImageRep(
    bitmapDataPlanes: nil,
    pixelsWide: Int(size.width),
    pixelsHigh: Int(size.height),
    bitsPerSample: 8,
    samplesPerPixel: 4,
    hasAlpha: true,
    isPlanar: false,
    colorSpaceName: .deviceRGB,
    bytesPerRow: 0,
    bitsPerPixel: 0
)!

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

let ctx = NSGraphicsContext.current!.cgContext

// Outer macOS squircle background (inset slightly for standard macOS icon padding)
let rect = CGRect(x: 92, y: 92, width: 840, height: 840)
let cornerRadius: CGFloat = 190
let bgPath = CGPath(roundedRect: rect, cornerWidth: cornerRadius, cornerHeight: cornerRadius, transform: nil)

// Subtle shadow
ctx.saveGState()
ctx.setShadow(offset: CGSize(width: 0, height: -14), blur: 32, color: NSColor.black.withAlphaComponent(0.45).cgColor)
ctx.addPath(bgPath)
ctx.setFillColor(NSColor(calibratedRed: 0.08, green: 0.10, blue: 0.15, alpha: 1.0).cgColor)
ctx.fillPath()
ctx.restoreGState()

// Clip to squircle and draw rich dark gradient
ctx.saveGState()
ctx.addPath(bgPath)
ctx.clip()

let colors = [
    NSColor(calibratedRed: 0.12, green: 0.16, blue: 0.28, alpha: 1.0).cgColor,
    NSColor(calibratedRed: 0.06, green: 0.07, blue: 0.12, alpha: 1.0).cgColor
] as CFArray
let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: colors, locations: [0.0, 1.0])!
ctx.drawLinearGradient(gradient, start: CGPoint(x: 512, y: 932), end: CGPoint(x: 512, y: 92), options: [])

// Subtle top-left accent glow
let glowColors = [
    NSColor(calibratedRed: 0.28, green: 0.56, blue: 0.98, alpha: 0.28).cgColor,
    NSColor.clear.cgColor
] as CFArray
let glow = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: glowColors, locations: [0.0, 1.0])!
ctx.drawRadialGradient(glow, startCenter: CGPoint(x: 320, y: 740), startRadius: 10, endCenter: CGPoint(x: 320, y: 740), endRadius: 540, options: [])

// Subtle inner border
ctx.addPath(bgPath)
ctx.setStrokeColor(NSColor.white.withAlphaComponent(0.14).cgColor)
ctx.setLineWidth(4)
ctx.strokePath()
ctx.restoreGState()

// Draw SF Symbol centerpiece
let symConfig = NSImage.SymbolConfiguration(pointSize: 380, weight: .medium)
if let symbol = NSImage(systemSymbolName: "arrow.triangle.2.circlepath.circle.fill", accessibilityDescription: nil)?.withSymbolConfiguration(symConfig) {
    let tinted = NSImage(size: symbol.size)
    tinted.lockFocus()
    let symRect = NSRect(origin: .zero, size: symbol.size)
    symbol.draw(in: symRect)
    NSColor(calibratedRed: 0.42, green: 0.72, blue: 1.0, alpha: 1.0).set()
    symRect.fill(using: .sourceAtop)
    tinted.unlockFocus()

    let drawSize = CGSize(width: 480, height: 480)
    let drawRect = CGRect(
        x: (size.width - drawSize.width) / 2,
        y: (size.height - drawSize.height) / 2 + 12,
        width: drawSize.width,
        height: drawSize.height
    )
    tinted.draw(in: drawRect, from: .zero, operation: .sourceOver, fraction: 1.0)
}

NSGraphicsContext.restoreGraphicsState()

let outURL = URL(fileURLWithPath: CommandLine.arguments[1])
let pngData = rep.representation(using: .png, properties: [:])!
try pngData.write(to: outURL)
print("Wrote icon PNG to \(outURL.path)")
