#pragma once

#include "ast_node.hh"
#include <string>

struct MODULE_NODE* parse_input_file(const std::string& file, const bool& debug);