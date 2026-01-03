/**
 * @file _alda_parser.cpp
 * @brief Python bindings for the Alda C parser via nanobind.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

extern "C" {
#include <alda/alda.h>
}

#include <memory>
#include <sstream>

namespace nb = nanobind;

/* Python wrapper for AldaToken */
struct PyToken {
    std::string type_name;
    std::string lexeme;
    int line;
    int column;
    nb::object literal;

    std::string repr() const {
        std::ostringstream oss;
        oss << "Token(" << type_name << ", '" << lexeme << "', "
            << line << ":" << column << ")";
        return oss.str();
    }
};

/* Python wrapper for AldaNode - recursive structure */
struct PyNode {
    std::string type_name;
    int line;
    int column;
    nb::dict data;
    std::vector<PyNode> children;

    std::string repr() const {
        std::ostringstream oss;
        oss << "ASTNode(" << type_name << " at " << line << ":" << column << ")";
        return oss.str();
    }
};

/* Convert C token to Python token */
static PyToken convert_token(const AldaToken& tok) {
    PyToken py_tok;
    py_tok.type_name = alda_token_type_name(tok.type);
    py_tok.lexeme = tok.lexeme ? tok.lexeme : "";
    py_tok.line = tok.pos.line;
    py_tok.column = tok.pos.column;

    /* Set literal based on token type */
    switch (tok.type) {
        case ALDA_TOK_NOTE_LENGTH:
        case ALDA_TOK_NOTE_LENGTH_MS:
        case ALDA_TOK_OCTAVE_SET:
        case ALDA_TOK_REPEAT:
            py_tok.literal = nb::int_(tok.literal.int_val);
            break;
        case ALDA_TOK_NOTE_LENGTH_S:
        case ALDA_TOK_NUMBER:
            py_tok.literal = nb::float_(tok.literal.float_val);
            break;
        case ALDA_TOK_NOTE_LETTER: {
            char buf[2] = {tok.literal.char_val, '\0'};
            py_tok.literal = nb::str(buf);
            break;
        }
        default:
            py_tok.literal = nb::none();
            break;
    }

    return py_tok;
}

/* Forward declaration */
static PyNode convert_node(AldaNode* node);

/* Convert linked list of nodes to vector */
static std::vector<PyNode> convert_node_list(AldaNode* list) {
    std::vector<PyNode> result;
    while (list) {
        result.push_back(convert_node(list));
        list = list->next;
    }
    return result;
}

/* Convert C AST node to Python node */
static PyNode convert_node(AldaNode* node) {
    PyNode py_node;
    py_node.type_name = alda_node_type_name(node->type);
    py_node.line = node->pos.line;
    py_node.column = node->pos.column;

    switch (node->type) {
        case ALDA_NODE_ROOT:
            py_node.children = convert_node_list(node->data.root.children);
            break;

        case ALDA_NODE_PART_DECL: {
            nb::list names;
            for (size_t i = 0; i < node->data.part_decl.name_count; i++) {
                names.append(nb::str(node->data.part_decl.names[i]));
            }
            py_node.data["names"] = names;
            if (node->data.part_decl.alias) {
                py_node.data["alias"] = nb::str(node->data.part_decl.alias);
            }
            break;
        }

        case ALDA_NODE_EVENT_SEQ:
            py_node.children = convert_node_list(node->data.event_seq.events);
            break;

        case ALDA_NODE_NOTE: {
            char letter_buf[2] = {node->data.note.letter, '\0'};
            py_node.data["letter"] = nb::str(letter_buf);
            if (node->data.note.accidentals) {
                py_node.data["accidentals"] = nb::str(node->data.note.accidentals);
            }
            py_node.data["slurred"] = nb::bool_(node->data.note.slurred != 0);
            if (node->data.note.duration) {
                py_node.children.push_back(convert_node(node->data.note.duration));
            }
            break;
        }

        case ALDA_NODE_REST:
            if (node->data.rest.duration) {
                py_node.children.push_back(convert_node(node->data.rest.duration));
            }
            break;

        case ALDA_NODE_CHORD:
            py_node.children = convert_node_list(node->data.chord.notes);
            break;

        case ALDA_NODE_DURATION:
            py_node.children = convert_node_list(node->data.duration.components);
            break;

        case ALDA_NODE_NOTE_LENGTH:
            py_node.data["denominator"] = nb::int_(node->data.note_length.denominator);
            py_node.data["dots"] = nb::int_(node->data.note_length.dots);
            break;

        case ALDA_NODE_NOTE_LENGTH_MS:
            py_node.data["ms"] = nb::int_(node->data.note_length_ms.ms);
            break;

        case ALDA_NODE_NOTE_LENGTH_S:
            py_node.data["seconds"] = nb::float_(node->data.note_length_s.seconds);
            break;

        case ALDA_NODE_OCTAVE_SET:
            py_node.data["octave"] = nb::int_(node->data.octave_set.octave);
            break;

        case ALDA_NODE_LISP_LIST:
            py_node.children = convert_node_list(node->data.lisp_list.elements);
            break;

        case ALDA_NODE_LISP_SYMBOL:
            py_node.data["name"] = nb::str(node->data.lisp_symbol.name);
            break;

        case ALDA_NODE_LISP_NUMBER:
            py_node.data["value"] = nb::float_(node->data.lisp_number.value);
            break;

        case ALDA_NODE_LISP_STRING:
            if (node->data.lisp_string.value) {
                py_node.data["value"] = nb::str(node->data.lisp_string.value);
            }
            break;

        case ALDA_NODE_VAR_DEF:
            py_node.data["name"] = nb::str(node->data.var_def.name);
            py_node.children = convert_node_list(node->data.var_def.events);
            break;

        case ALDA_NODE_VAR_REF:
            py_node.data["name"] = nb::str(node->data.var_ref.name);
            break;

        case ALDA_NODE_MARKER:
            py_node.data["name"] = nb::str(node->data.marker.name);
            break;

        case ALDA_NODE_AT_MARKER:
            py_node.data["name"] = nb::str(node->data.at_marker.name);
            break;

        case ALDA_NODE_VOICE_GROUP:
            py_node.children = convert_node_list(node->data.voice_group.voices);
            break;

        case ALDA_NODE_VOICE:
            py_node.data["number"] = nb::int_(node->data.voice.number);
            py_node.children = convert_node_list(node->data.voice.events);
            break;

        case ALDA_NODE_CRAM:
            py_node.children = convert_node_list(node->data.cram.events);
            if (node->data.cram.duration) {
                /* Add duration as special child */
                py_node.data["duration"] = convert_node(node->data.cram.duration).data;
            }
            break;

        case ALDA_NODE_BRACKET_SEQ:
            py_node.children = convert_node_list(node->data.bracket_seq.events);
            break;

        case ALDA_NODE_REPEAT:
            py_node.data["count"] = nb::int_(node->data.repeat.count);
            if (node->data.repeat.event) {
                py_node.children.push_back(convert_node(node->data.repeat.event));
            }
            break;

        case ALDA_NODE_ON_REPS:
            if (node->data.on_reps.event) {
                py_node.children.push_back(convert_node(node->data.on_reps.event));
            }
            break;

        default:
            break;
    }

    return py_node;
}

/* Scan source and return tokens */
static std::vector<PyToken> py_scan(const std::string& source, const std::string& filename) {
    AldaScanner* scanner = alda_scanner_new(source.c_str(), filename.c_str());
    if (!scanner) {
        throw std::runtime_error("Failed to create scanner");
    }

    size_t count = 0;
    AldaToken* tokens = alda_scanner_scan(scanner, &count);

    if (alda_scanner_has_error(scanner)) {
        char* err_str = alda_scanner_error_string(scanner);
        std::string err_msg = err_str ? err_str : "Scan error";
        free(err_str);
        alda_scanner_free(scanner);
        throw std::runtime_error(err_msg);
    }

    alda_scanner_free(scanner);

    std::vector<PyToken> result;
    for (size_t i = 0; i < count; i++) {
        result.push_back(convert_token(tokens[i]));
    }

    alda_tokens_free(tokens, count);
    return result;
}

/* Parse source and return AST */
static PyNode py_parse(const std::string& source, const std::string& filename) {
    char* error = nullptr;
    AldaNode* ast = alda_parse(source.c_str(), filename.c_str(), &error);

    if (!ast) {
        std::string err_msg = error ? error : "Parse error";
        free(error);
        throw std::runtime_error(err_msg);
    }

    PyNode result = convert_node(ast);
    alda_ast_free(ast);
    return result;
}

NB_MODULE(_alda_parser, m) {
    m.doc() = "Alda C parser Python bindings";

    /* Token class */
    nb::class_<PyToken>(m, "Token")
        .def_ro("type", &PyToken::type_name, "Token type name")
        .def_ro("lexeme", &PyToken::lexeme, "Token text")
        .def_ro("line", &PyToken::line, "Line number (1-based)")
        .def_ro("column", &PyToken::column, "Column number (1-based)")
        .def_ro("literal", &PyToken::literal, "Parsed literal value")
        .def("__repr__", &PyToken::repr);

    /* AST Node class */
    nb::class_<PyNode>(m, "ASTNode")
        .def_ro("type", &PyNode::type_name, "Node type name")
        .def_ro("line", &PyNode::line, "Line number (1-based)")
        .def_ro("column", &PyNode::column, "Column number (1-based)")
        .def_ro("data", &PyNode::data, "Node-specific data")
        .def_ro("children", &PyNode::children, "Child nodes")
        .def("__repr__", &PyNode::repr);

    /* Module functions */
    m.def("scan", &py_scan, nb::arg("source"), nb::arg("filename") = "<input>",
          "Scan source text and return a list of tokens");

    m.def("parse", &py_parse, nb::arg("source"), nb::arg("filename") = "<input>",
          "Parse source text and return an AST");

    m.def("get_version", []() { return "0.1.0"; },
          "Get the parser version");
}
