import SwiftUI
import WebKit
import AppKit

struct ReadmeWebView: NSViewRepresentable {
    let markdown: String
    let repoId: String
    @Binding var contentHeight: CGFloat

    func makeCoordinator() -> Coordinator {
        Coordinator(contentHeight: $contentHeight)
    }

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        webView.setValue(false, forKey: "drawsBackground")
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        let key = "\(repoId)|\(markdown.hashValue)"
        guard context.coordinator.lastLoadedKey != key else { return }
        context.coordinator.lastLoadedKey = key
        let html = Self.buildHTML(markdown: markdown, repoId: repoId)
        let baseURL = URL(string: "https://huggingface.co/\(repoId)/resolve/main/")
        webView.loadHTMLString(html, baseURL: baseURL)
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        @Binding var contentHeight: CGFloat
        var lastLoadedKey: String = ""

        init(contentHeight: Binding<CGFloat>) {
            _contentHeight = contentHeight
        }

        @MainActor
        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction
        ) async -> WKNavigationActionPolicy {
            if navigationAction.navigationType == .linkActivated,
               let url = navigationAction.request.url {
                NSWorkspace.shared.open(url)
                return .cancel
            }
            return .allow
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            webView.evaluateJavaScript("document.body.scrollHeight") { result, _ in
                if let h = result as? CGFloat, h > 20 {
                    DispatchQueue.main.async {
                        self.contentHeight = min(max(h + 16, 120), 4000)
                    }
                }
            }
        }
    }

    // MARK: - Markdown + HTML Rendering

    private static func buildHTML(markdown: String, repoId: String) -> String {
        let body = markdownToHTML(String(markdown.prefix(40_000)))
        return """
        <!doctype html>
        <html>
        <head>
        <meta charset="utf-8">
        <meta name="color-scheme" content="light dark">
        <style>
          :root {
            color-scheme: light dark;
          }
          html, body {
            margin: 0;
            padding: 4px 6px;
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
            font-size: 13px;
            line-height: 1.55;
            color: -apple-system-label;
            word-wrap: break-word;
            overflow-wrap: anywhere;
          }
          @media (prefers-color-scheme: dark) {
            body { color: #e5e5ea; }
            a { color: #58a6ff; }
            code, pre { background: rgba(255,255,255,0.08); }
            blockquote { border-left-color: rgba(255,255,255,0.25); color: #aeaeb2; }
            th, td { border-color: rgba(255,255,255,0.16); }
            hr { border-top-color: rgba(255,255,255,0.16); }
          }
          @media (prefers-color-scheme: light) {
            body { color: #1c1c1e; }
            a { color: #0969da; }
            code, pre { background: rgba(0,0,0,0.05); }
            blockquote { border-left-color: rgba(0,0,0,0.2); color: #636366; }
            th, td { border-color: rgba(0,0,0,0.15); }
            hr { border-top-color: rgba(0,0,0,0.15); }
          }
          h1, h2, h3, h4, h5, h6 {
            margin: 14px 0 6px;
            font-weight: 600;
            line-height: 1.25;
          }
          h1 { font-size: 1.45em; }
          h2 { font-size: 1.25em; }
          h3 { font-size: 1.1em; }
          p { margin: 8px 0; }
          ul, ol { margin: 6px 0; padding-left: 20px; }
          li { margin: 3px 0; }
          code {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.9em;
            padding: 1px 4px;
            border-radius: 4px;
          }
          pre {
            padding: 10px 12px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 8px 0;
          }
          pre code {
            background: transparent;
            padding: 0;
          }
          blockquote {
            margin: 8px 0;
            padding: 2px 12px;
            border-left: 3px solid;
          }
          table {
            border-collapse: collapse;
            margin: 10px 0;
            width: auto;
            max-width: 100%;
            display: block;
            overflow-x: auto;
          }
          th, td {
            border: 1px solid;
            padding: 5px 10px;
            text-align: left;
          }
          img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
          }
          hr {
            border: none;
            border-top: 1px solid;
            margin: 12px 0;
          }
          a { text-decoration: none; }
          a:hover { text-decoration: underline; }
        </style>
        </head>
        <body>\(body)</body>
        </html>
        """
    }

    private static func markdownToHTML(_ input: String) -> String {
        let lines = input.components(separatedBy: .newlines)
        var out: [String] = []
        var i = 0
        var inCodeBlock = false
        var codeLines: [String] = []
        var listType: String? = nil

        func closeList() {
            if let lt = listType {
                out.append("</\(lt)>")
                listType = nil
            }
        }

        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Fenced code blocks
            if trimmed.hasPrefix("```") {
                closeList()
                if inCodeBlock {
                    let escaped = escapeHTML(codeLines.joined(separator: "\n"))
                    out.append("<pre><code>\(escaped)</code></pre>")
                    codeLines.removeAll()
                    inCodeBlock = false
                } else {
                    inCodeBlock = true
                }
                i += 1
                continue
            }

            if inCodeBlock {
                codeLines.append(line)
                i += 1
                continue
            }

            if trimmed.isEmpty {
                closeList()
                i += 1
                continue
            }

            // Horizontal rule
            if trimmed == "---" || trimmed == "***" || trimmed == "___" {
                closeList()
                out.append("<hr>")
                i += 1
                continue
            }

            // Headings
            if let heading = parseHeading(trimmed) {
                closeList()
                out.append("<h\(heading.level)>\(renderInline(heading.text))</h\(heading.level)>")
                i += 1
                continue
            }

            // Blockquote
            if trimmed.hasPrefix(">") {
                closeList()
                let quoteText = String(trimmed.dropFirst()).trimmingCharacters(in: .whitespaces)
                out.append("<blockquote>\(renderInline(quoteText))</blockquote>")
                i += 1
                continue
            }

            // Markdown table
            if trimmed.hasPrefix("|"), i + 1 < lines.count, isTableSeparator(lines[i + 1]) {
                closeList()
                let headers = splitTableRow(trimmed)
                var tableHTML = "<table><thead><tr>"
                for h in headers {
                    tableHTML += "<th>\(renderInline(h))</th>"
                }
                tableHTML += "</tr></thead><tbody>"
                i += 2
                while i < lines.count {
                    let rowTrimmed = lines[i].trimmingCharacters(in: .whitespaces)
                    guard rowTrimmed.hasPrefix("|") else { break }
                    let cols = splitTableRow(rowTrimmed)
                    tableHTML += "<tr>"
                    for c in cols {
                        tableHTML += "<td>\(renderInline(c))</td>"
                    }
                    tableHTML += "</tr>"
                    i += 1
                }
                tableHTML += "</tbody></table>"
                out.append(tableHTML)
                continue
            }

            // Unordered list
            if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") || trimmed.hasPrefix("+ ") {
                if listType != "ul" {
                    closeList()
                    out.append("<ul>")
                    listType = "ul"
                }
                let item = String(trimmed.dropFirst(2))
                out.append("<li>\(renderInline(item))</li>")
                i += 1
                continue
            }

            // Ordered list
            if let olRange = trimmed.range(of: #"^\d+\.\s+"#, options: .regularExpression) {
                if listType != "ol" {
                    closeList()
                    out.append("<ol>")
                    listType = "ol"
                }
                let item = String(trimmed[olRange.upperBound...])
                out.append("<li>\(renderInline(item))</li>")
                i += 1
                continue
            }

            closeList()

            // Pass through block-level HTML lines directly, otherwise wrap in <p>
            if trimmed.hasPrefix("<") && !trimmed.hasPrefix("<a ") && !trimmed.hasPrefix("<b") && !trimmed.hasPrefix("<i") && !trimmed.hasPrefix("<small") && !trimmed.hasPrefix("<span") && !trimmed.hasPrefix("<strong") && !trimmed.hasPrefix("<em") && !trimmed.hasPrefix("<code") {
                out.append(line)
            } else {
                out.append("<p>\(renderInline(line))</p>")
            }
            i += 1
        }

        if inCodeBlock {
            let escaped = escapeHTML(codeLines.joined(separator: "\n"))
            out.append("<pre><code>\(escaped)</code></pre>")
        }
        closeList()
        return out.joined(separator: "\n")
    }

    private static func parseHeading(_ line: String) -> (level: Int, text: String)? {
        var level = 0
        for ch in line {
            if ch == "#" { level += 1 } else { break }
        }
        guard (1...6).contains(level) else { return nil }
        let rest = line.dropFirst(level)
        guard rest.first == " " else { return nil }
        return (level, String(rest.dropFirst()).trimmingCharacters(in: .whitespaces))
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        let t = line.trimmingCharacters(in: .whitespaces)
        return t.hasPrefix("|") && t.contains("-")
    }

    private static func splitTableRow(_ line: String) -> [String] {
        var s = line.trimmingCharacters(in: .whitespaces)
        if s.hasPrefix("|") { s.removeFirst() }
        if s.hasSuffix("|") { s.removeLast() }
        return s.components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
    }

    private static func renderInline(_ text: String) -> String {
        var s = text
        // Inline code: `code`
        s = replaceRegex(s, pattern: #"`([^`]+)`"#) { match in
            "<code>\(escapeHTML(match[1]))</code>"
        }
        // Images: ![alt](url)
        s = replaceRegex(s, pattern: #"!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)"#) { match in
            "<img src=\"\(match[2])\" alt=\"\(escapeHTML(match[1]))\">"
        }
        // Links: [text](url)
        s = replaceRegex(s, pattern: #"\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)"#) { match in
            "<a href=\"\(match[2])\">\(match[1])</a>"
        }
        // Bold: **text** or __text__
        s = replaceRegex(s, pattern: #"\*\*([^\*]+)\*\*"#) { "<strong>\($0[1])</strong>" }
        s = replaceRegex(s, pattern: #"__([^_]+)__"#) { "<strong>\($0[1])</strong>" }
        // Italic: *text*
        s = replaceRegex(s, pattern: #"(?<!\*)\*([^\*]+)\*(?!\*)"#) { "<em>\($0[1])</em>" }
        return s
    }

    private static func replaceRegex(_ input: String, pattern: String, transform: ([String]) -> String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return input }
        let ns = input as NSString
        let matches = regex.matches(in: input, range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else { return input }
        var result = input
        for m in matches.reversed() {
            var groups: [String] = []
            for idx in 0..<m.numberOfRanges {
                let r = m.range(at: idx)
                groups.append(r.location != NSNotFound ? ns.substring(with: r) : "")
            }
            if let swiftRange = Range(m.range, in: result) {
                result.replaceSubrange(swiftRange, with: transform(groups))
            }
        }
        return result
    }

    private static func escapeHTML(_ text: String) -> String {
        text
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }
}
