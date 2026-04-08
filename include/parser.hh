#pragma once

#include "ptoNode.hh"
#include <string>

pto_parser::PTO_MODULE* parse_input_file(const std::string& file, const bool& debug);