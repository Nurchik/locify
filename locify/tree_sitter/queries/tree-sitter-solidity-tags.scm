(contract_declaration
  name: (identifier) @name.definition.contract) @definition.contract

(interface_declaration
  name: (identifier) @name.definition.interface) @definition.interface

(library_declaration
  name: (identifier) @name.definition.library) @definition.library

(function_definition
  name: (identifier) @name.definition.function) @definition.function

(struct_declaration
  name: (identifier) @name.definition.struct) @definition.struct

(error_declaration
  name: (identifier) @name.definition.error) @definition.error

(event_definition
  name: (identifier) @name.definition.event) @definition.event

(modifier_definition
  name: (identifier) @name.definition.modifier) @definition.modifier

(state_variable_declaration
  name: (identifier) @name.definition.state_variable) @definition.state_variable

(user_defined_type_definition
  name: (identifier) @name.definition.user_type) @definition.user_type

(call_expression
  [
    (expression (identifier) @name.reference.call)
    (expression
      (member_expression 
        property: (identifier) @name.reference.method))
  ]
) @reference.call