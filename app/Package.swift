// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ModelBridge",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "ModelBridge", targets: ["ModelBridge"])
    ],
    targets: [
        .executableTarget(
            name: "ModelBridge",
            path: "Sources/ModelBridge"
        )
    ]
)
