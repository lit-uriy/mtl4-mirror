proc treenode {name params body} {
  set tparams ""; set decls ""; set consparams ""; set inits ""
  set makerargs ""; set makerargs2 ""; set makerargs2_t ""
  foreach tpraw $params {
    regsub {^-} $tpraw "" tp
    set tpl [string tolower $tp]
    set tpl_ "${tpl}_"
    lappend tparams "class $tp"
    lappend decls "$tp $tpl;"
    lappend consparams "const $tp& $tpl_"
    lappend inits "${tpl}($tpl_)"
    lappend makerargs "$tpl_"
    if {[string match {-*} $tpraw]} {
      lappend makerargs2 "$tpl_"
      lappend makerargs2_t "${tp}()"
    } else {
      lappend makerargs2 "wrap($tpl_)"
      lappend makerargs2_t "wrap(${tp}())"
    }
  }
  set tparams [join $tparams ", "]
  set decls [join $decls " "]
  set consparams [join $consparams ", "]
  set inits [join $inits ", "]
  set makerargs [join $makerargs ", "]
  set makerargs2 [join $makerargs2 ", "]
  set makerargs2_t [join $makerargs2_t ", "]
  regsub -all -- - $params "" tpnames
  set tpnames [join $tpnames ", "]

  return [subst {
template <$tparams>
struct ${name}_type: public expr_node {
  $decls
  ${name}_type() {}
  ${name}_type($consparams): $inits {}
  [subst $body]
};
template <$tparams>
${name}_type<$tpnames>
${name}_maker($consparams) {
  return ${name}_type<$tpnames>($makerargs);
}
template <$tparams>
typeof(${name}_maker($makerargs2_t))
${name}($consparams) {
  return ${name}_maker($makerargs2);
}
}]}

proc eval_is {ret args} {
  return [subst {
  template <class Env>
  typeof($ret) run(const Env& env) {
    [join [lrange $args 0 end-1] "\n"]
    return ([lindex $args end]);
  }
}]}
