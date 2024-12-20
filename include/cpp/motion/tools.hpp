/*!
 * \file
 * \brief C++ system and conversion functions.
 */

#pragma once

#include <vector>
#include <motion/tools.h>

/**
 * Convert int cvector into int std::vector.
 * @param arg Input int cvector.
 * @return Data converted in int std::vector.
 */
std::vector<std::size_t> tools_convert_int_cvector_int_stdvector(const vec_int_t arg);

/**
 * Convert int cvector into bool std::vector.
 * @param arg Input int cvector.
 * @return Data converted in bool std::vector.
 */
std::vector<bool> tools_convert_int_cvector_bool_stdvector(const vec_int_t arg);

/**
 * Convert int vector2D into std::vector<std::vector<int>>.
 * @param arg Input int matrix.
 * @return Data converted in std::vector<std::vector<int>>.
 */
std::vector<std::vector<std::size_t>> tools_convert_int_cvector2D_int_stdvector2D(const vec2D_int_t arg);
