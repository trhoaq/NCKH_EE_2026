set(RUST_TARGET "xtensa-esp32s3-espidf")
set(RUST_CRATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../rust/wakeword_bridge)
set(RUST_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/cargo-target)
set(RUST_STATICLIB ${RUST_TARGET_DIR}/${RUST_TARGET}/release/libwakeword_bridge.a)

add_custom_command(
        OUTPUT ${RUST_STATICLIB}
        COMMAND ${CMAKE_COMMAND} -E env
        CARGO_TARGET_DIR=${RUST_TARGET_DIR}
        IDF_PATH=$ENV{IDF_PATH}
        cargo build --manifest-path ${RUST_CRATE_DIR}/Cargo.toml --target ${RUST_TARGET} --release
        WORKING_DIRECTORY ${RUST_CRATE_DIR}
        DEPENDS
        ${RUST_CRATE_DIR}/Cargo.toml
        ${RUST_CRATE_DIR}/build.rs
        ${RUST_CRATE_DIR}/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../model/wakewords.manifest
        VERBATIM
)

add_custom_target(rust_bridge_staticlib ALL DEPENDS ${RUST_STATICLIB})

add_library(rust_bridge_prebuilt STATIC IMPORTED GLOBAL)
set_target_properties(rust_bridge_prebuilt PROPERTIES IMPORTED_LOCATION ${RUST_STATICLIB})
add_dependencies(rust_bridge_prebuilt rust_bridge_staticlib)

idf_component_register(
        SRCS "rust_bridge_shim.c"
        INCLUDE_DIRS "include"
)

target_link_libraries(${COMPONENT_LIB} PUBLIC rust_bridge_prebuilt)
