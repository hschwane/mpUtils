/*
 * mpUtils
 * ArgParser.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the ArgParser class
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_ARGPARSER_H
#define MPUTILS_ARGPARSER_H

// includes
//--------------------
#include <variant>
#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>
#include <cstdlib>

#include "mpUtils/IO/readData.h"
#include "mpUtils/Misc/stringUtils.h"
//--------------------

// settings
//--------------------
#define ARGPARSER_DEBUG_OUT std::cout
//#define ARGPARSER_DEBUG_OUT if(false) std::cout
//--------------------

// namespace
//--------------------
namespace mpu {
namespace cfg {
//--------------------

namespace detail { struct PositionalOnly { }; }

template <class OptionStruct = detail::PositionalOnly, bool allowPositional = true>
struct ArgParser
{
    /**
     * @brief enum to store one option
     */
    using Option_t = std::variant<std::string OptionStruct::*, int OptionStruct::*, float OptionStruct::*,
            double OptionStruct::*, bool OptionStruct::*>;

    /**
     * @brief struct stores information for the parser to parse a single argument
     */
    struct ArgInfo{
        std::string longName{};
        std::string shortName{};
        Option_t opt{};
        std::string description{};
        std::string valueDesc{};
    };

    /**
     * @brief this adds a vector of positional arguments to the parsers output data
     * @tparam os the struct to add positional data to
     * @tparam ap whether or not positional arguments need to be stored
     */
    template <class os, bool ap>
    struct Options : public os
    {
        std::vector<std::string> positional{};
    };

    template <class os>
    struct Options<os, false> : public os
    {
    };

    using Options_t = Options<OptionStruct,allowPositional>;

    /**
     * @brief Function that parse command line arguments into an option structure.
     *          Set your option structure as the template argument "OptionStruct" to the class, set default values
     *          for options using the default constructor of option struct. Set "allowPositional"
     *          if you want to allow positional arguments. Call "parse" with a vector of "ArgInfo". each "ArgInfo"
     *          contains info for one option. A long name, a short (one char) name, the adress of one of "OptionStruct"s
     *          members, a description for the argument, a discription for the value that is displayed next to
     *          the name (eg --input <file>). Do NOT include "--" or "-" into the long or short name!
     *
     *          Arguments are passed as "--longName", or "-shortName" you can then either
     *          use a space or "=" before passing the value. Bool arguments that appear without a value are set to true.
     *          Use \@ sometextfile.txt to load arguments from a text file
     *
     *          Automatically handles "-h", "--help", and "--version", exiting the application with exit code 0 after
     *          the info was printed. Throws an illegal argument exception and prints error to output stream in case
     *          of a parsing error.
     *
     * @param argc number of commandline arguments
     * @param argv commandline arguments
     * @param args vector or "Arg" defining possible
     * @param usageText text on top of the options list when "-h" or "--help" is passed
     * @param versionText the text displayed when --version is passed
     * @param output ostream to which errors or help messages are written
     * @return the options struct appended by vector of positional arguments if allowPositional was set
     */
    static Options_t parse(int argc, const char* argv[], const std::vector<ArgInfo>& args, std::string_view usageText,
                    std::string_view versionText, std::ostream &output = std::cout);
private:
    static void printHelp(const std::vector<ArgInfo>& args, std::string_view usageText, std::ostream &output);
    static bool parseOption(Options_t& options, const ArgParser::Option_t& opt, const std::string& value,
                            std::ostream& output, std::string_view name);
    static std::string_view parseRecursive(const std::vector<std::string_view>& tokens, int tid, Options_t& options,
                                           const std::vector<ArgInfo>& args, std::string_view usageText,
                                           std::string_view versionText, std::ostream& output);
    static void handlePositional(Options_t& options, std::string_view value, std::ostream& output);
};

//-------------------------------------------------------------------
// template function definitions

template <class OptionStruct, bool allowPositional>
typename ArgParser<OptionStruct,allowPositional>::Options_t
ArgParser<OptionStruct, allowPositional>::parse(int argc, const char* argv[],
                                                const std::vector<ArgInfo>& args,
                                                std::string_view usageText,
                                                std::string_view versionText,
                                                std::ostream &output)
{
    // create options struct with defaults
    Options_t options{};

    if(argc > 1) {
        // make tokens from input arguments, also split options that start with at least one '-' at '='
        std::vector<std::string_view> tokens;
        for(int i=1; i<argc; ++i) {
            std::string_view s(argv[i]);
            if(s[0] == '-') {
                auto v = tokenize(s,'=');
                tokens.insert(tokens.end(), v.begin(), v.end());
            } else
                tokens.push_back(s);
        }

        // log tokens for debugging
        ARGPARSER_DEBUG_OUT << "tokenizing arguments: \n";
        for(auto&& t : tokens)
            ARGPARSER_DEBUG_OUT << "\t" << t << "\n";

        // parse using recursive function
        std::string_view value = parseRecursive(tokens,0,options,args,usageText,versionText,output);
        if(!value.empty())
            handlePositional(options,value,output);
    }

    return options;
}

template <class OptionStruct, bool allowPositional>
bool ArgParser<OptionStruct, allowPositional>::parseOption(Options_t& options, const ArgParser::Option_t& opt,
                                                             const std::string& value, std::ostream& output,
                                                             std::string_view name)
{
    if(std::holds_alternative<bool OptionStruct::*>(opt)) {
        // for boolean simply set true
        ARGPARSER_DEBUG_OUT << "Found boolean arg: " << name << ".\n";
        options.*std::get<bool OptionStruct::*>(opt) = true;
        return false;
    } else {
        ARGPARSER_DEBUG_OUT << "Found arg: " << name << ".\n";
        // if something else, see if we have a parsed value
        if(value.empty()) {
            output << "Missing value for option \"" << name
                   << "\".\nUse --help for help.\n";
            throw std::invalid_argument(std::string("Missing value for option \"") + std::string(name) + "\".");
        }
        try {
            if(std::holds_alternative<std::string OptionStruct::*>(opt))
                options.*std::get<std::string OptionStruct::*>(opt) = std::string(value);
            else if(std::holds_alternative<int OptionStruct::*>(opt))
                options.*std::get<int OptionStruct::*>(opt) = std::stoi(value);
            else if(std::holds_alternative<float OptionStruct::*>(opt))
                options.*std::get<float OptionStruct::*>(opt) = std::stof(value);
            else if(std::holds_alternative<double OptionStruct::*>(opt))
                options.*std::get<double OptionStruct::*>(opt) = std::stod(value);
            else
                throw std::logic_error("Missing if branch!");
        } catch(std::exception& e) {
            output << "Invalid value for option \"" << name << "\".\nUse --help for help.\n";
            throw std::invalid_argument(std::string("Invalid value for option  \"") + std::string(name) + "\".");
        }
        return true;
    }
}

template <class OptionStruct, bool allowPositional>
void ArgParser<OptionStruct, allowPositional>::handlePositional(ArgParser::Options_t& options, std::string_view value,
                                                                std::ostream& output)
{
    ARGPARSER_DEBUG_OUT << "Dealing with positional. \n";
    if constexpr (allowPositional) {
        options.positional.emplace_back(value);
    } else {
        output << "Unknown option \"" << value << "\".\nUse --help for help.\n";
        throw std::invalid_argument(std::string("unknown option \"") + std::string(value) + "\".");
    }
}

template <class OptionStruct, bool allowPositional>
std::string_view
ArgParser<OptionStruct, allowPositional>::parseRecursive(const std::vector<std::string_view>& tokens,
                                                         int tid,
                                                         Options_t& options,
                                                         const std::vector<ArgInfo>& args,
                                                         std::string_view usageText,
                                                         std::string_view versionText,
                                                         std::ostream& output)
{
    // recursion stop condition
    if(tid >= tokens.size())
        return std::string_view();

    std::string_view token = tokens[tid];
    if(token.empty()) {
        ARGPARSER_DEBUG_OUT << "Ignoring empty token. \n";
        return parseRecursive(tokens, tid+1, options, args,
                              usageText, versionText, output);
    }

    if(token[0] == '-') {
        if(token[1] == '-') {
            // long arguments
            std::string_view subtoken = token.substr(2);
            if(subtoken == "help") {
                printHelp(args,usageText,output);
                std::exit(0);
            } else if(subtoken == "version") {
                output << versionText << "\n";
                std::exit(0);
            }

            // find matching ArgumentInfo
            for(auto&& arg : args) {
                if(!arg.longName.empty() && arg.longName == subtoken) {
                    // parse the next token to get input value
                    std::string_view value = parseRecursive(tokens, tid+1, options, args,
                                                            usageText, versionText, output);
                    // se if the value fits the token and assign it
                    if(parseOption(options, arg.opt, std::string(value), output, subtoken))
                        return std::string_view();
                    else
                        return value;
                }
            }
            // option was not found
            output << "Unknown option \"" << subtoken << "\".\nUse --help for help.\n";
            throw std::invalid_argument(std::string("unknown option \"") + std::string(subtoken) + "\".");

        } else {
            // short argument
            // check every char individually, to a allow "-abc"
            for(int i=1; i<token.length(); ++i) {
                std::string_view s = token.substr(i,1);
                if(s == "h") {
                    printHelp(args,usageText,output);
                    std::exit(0);
                }

                // find matching ArgumentInfo
                bool found = false;
                for(auto&& arg : args) {
                    if(!arg.shortName.empty() && arg.shortName == s) {
                        found = true;
                        // see if the value fits the token and assign it
                        if(i==token.length()-1) {
                            // parse the next token, maybe we need an input value
                            std::string_view value = parseRecursive(tokens, tid + 1, options, args, usageText, versionText, output);
                            if(parseOption(options, arg.opt, std::string(value), output, s))
                                return std::string_view();
                            else
                                return value;
                        } else {
                            parseOption(options, arg.opt, std::string(), output, s);
                            break;
                        }
                    }
                }
                if(!found) {
                    output << "Unknown option \"-" << s << "\".\nUse --help for help.\n";
                    throw std::invalid_argument(std::string("unknown option \"") + std::string(s) + "\".");
                }
            }
        }
    } else if(token[0] == '@') {
        // read arguments from file
        ARGPARSER_DEBUG_OUT << "Reading arguments from file. \n";
        std::vector<std::string_view> fileTokens;
        std::string fileContent = readFile(std::string(token.substr(1)));
        std::vector<std::string_view> fileTokensRaw = tokenize(std::string_view(fileContent));
        fileTokens.reserve(fileTokensRaw.size());
        for(auto& it : fileTokensRaw) {
            if(it[0] == '-') {
                auto v = tokenize(it, '=');
                fileTokens.insert(fileTokens.end(), v.begin(), v.end());
            } else
                fileTokens.push_back(it);
        }
        ARGPARSER_DEBUG_OUT << "tokenizing file:\n";
        for(auto&& t : fileTokensRaw)
            ARGPARSER_DEBUG_OUT << "\t" << t << "\n";

        std::string_view value = parseRecursive(fileTokens, 0, options, args,
                                        usageText, versionText, output);
        if(!value.empty())
            handlePositional(options, value, output);

        // continue parsing
        return parseRecursive(tokens, tid+1, options, args,
                              usageText, versionText, output);
    } else {
        // continue parsing
        std::string_view value = parseRecursive(tokens, tid+1, options, args,
                                                usageText, versionText, output);
        if(!value.empty())
            handlePositional(options, value, output);

        // setting or positional argument
        return token;
    }
}

template <class OptionStruct, bool allowPositional>
    void mpu::cfg::ArgParser<OptionStruct, allowPositional>::printHelp(const std::vector<ArgInfo>& args,
                                                                       std::string_view usageText, std::ostream& output)
{
    if(!usageText.empty())
        output << "Usage: " << usageText << "\n";

    output << "Options:\n";
    for(const auto& arg : args) {
        std::ostringstream ss;

        ss << "  ";
        if(!arg.shortName.empty())
            ss << "-" << arg.shortName;
        if(!arg.longName.empty() && !arg.shortName.empty())
            ss << ", ";
        if(!arg.longName.empty())
            ss << "--" << arg.longName;
        if(!arg.valueDesc.empty())
            ss << " " << arg.valueDesc;
        output << std::left << std::setw(36) << ss.str() << "  " << arg.description << "\n";
    }
    output << "\nValues can be set by \"--arg v\", \"--arg=v\" or \"-a v\", \"-a=v\".\n"
              "Single letter switches can be combined (\"-a -b\" <=> \"-ab\").\n"
              "Use @<textfile.txt> to read arguments from a text file.";
}

}}
#endif //MPUTILS_ARGPARSER_H