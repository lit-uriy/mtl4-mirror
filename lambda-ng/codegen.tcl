proc treenode {name params body} {
  set tparams ""; set decls ""; set consparams ""; set inits ""
  set makerargs ""; set makerargs2 ""
  foreach tp $params {
    set tpl [string tolower $tp]
    set tpl_ "${tpl}_"
    lappend tparams "class $tp"
    lappend decls "$tp $tpl;"
    lappend consparams "const $tp& $tpl_"
    lappend inits "${tpl}($tpl_)"
    lappend makerargs "$tpl_"
    lappend makerargs2 "wrap($tpl_)"
  }
  set tparams [join $tparams ", "]
  set decls [join $decls " "]
  set consparams [join $consparams ", "]
  set inits [join $inits ", "]
  set makerargs [join $makerargs ", "]
  set makerargs2 [join $makerargs2 ", "]
  return [subst {
template <$tparams>
struct ${name}_type: public expr_node {
  $decls
  ${name}_type($consparams): $inits {}
  [subst $body]
};
template <$tparams>
${name}_type<[join $params ", "]>
${name}_maker($consparams) {
  return ${name}_type<[join $params ", "]>($makerargs);
}
template <$tparams>
typeof(${name}_maker($makerargs2))
${name}($consparams) {
  return ${name}_maker($makerargs2);
}
}]}

proc eval_is {args} {
  return [subst {
  template <class Env>
  typeof([lindex $args end]) run(const Env& env) {
    [join [lrange $args 0 end-1] "\n"]
    return ([lindex $args end]);
  }
}]}
