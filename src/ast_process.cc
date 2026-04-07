#include "ast_process.hh"
#include "logger.hh"

void ast_class_delete(struct CLASS_NODE *c) {
    if (c == nullptr) return;


    delete c;
}

void ast_module_delete(struct MODULE_NODE *module) {
    if (module == nullptr) return;



    delete module;
}