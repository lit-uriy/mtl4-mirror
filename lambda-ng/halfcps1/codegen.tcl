#!/usr/bin/tclsh

proc getfile {fn} {
  set f [open $fn r]
  set result [read $f]
  close $f
  return $result
}

proc putfile {fn data} {
  set f [open $fn w]
  puts -nonewline $f $data
  close $f
}

proc closure {name memtypes tparams contentstr} {
  set is_top [regsub -all ! $name "" name];
  # Sets is_top to 1 if there was a !, and removes it from name
  set tpdecls ""; set members ""; set constargs ""; set setters ""
  set tpl_s ""; set tpnames ""
  if {!$is_top} {
    regsub -all {\^} $memtypes "" memtypes
  }
  foreach tp $tparams {
    lappend tpdecls "class $tp"
    lappend tpnames "$tp"
  }
  foreach tp $memtypes {
    set tp_old $tp
    set is_carat [regsub -all {\^} $tp "" tp]
    set tpl [string tolower $tp]
    set tpl_ ${tpl}_
    lappend tpdecls "class $tp"
    lappend members "$tp $tpl;"
    lappend constargs "const $tp& $tpl_"
    lappend setters "${tpl}($tpl_)"
    if {$is_top && !$is_carat} {
      lappend tpnames "typename node_traits<$tp>::wrapped"
      lappend tpl_s wrap($tpl_)
    } else {
      lappend tpnames "$tp"
      lappend tpl_s $tpl_
    }
  }
  set tpl_s [join $tpl_s ", "]
  set tpdecls [join $tpdecls ", "]
  set members [join $members " "]
  set constargs [join $constargs ", "]
  set setters [join $setters ", "]
  set tpnames [join $tpnames ", "]
  if {$is_top} {
    set inh ": public expr_node"
  } else {
    set inh ""
  }
  return [subst {
  template <$tpdecls>
  struct ${name}_type$inh {
    $members
    ${name}_type($constargs): $setters {}
    $contentstr
  };
  template <$tpdecls>
  ${name}_type<$tpnames>
  ${name}($constargs) {
    return ${name}_type<$tpnames>($tpl_s);
  }
  }]
}

proc get_type_is {args} {
  return [list get_type $args]
}

proc returns {t} {
  return [list returns $t]
}

proc run_is {content} {
  return [list run $content]
}

proc typed_closure {name memtypes stored_types args} {
  foreach keyval $args {
    set content([lindex $keyval 0]) [lindex $keyval 1]
  }
  set st_tp ""; set st_mt ""
  foreach tp $stored_types {
    if {[regsub -all ^% $tp "" tp]} {
      # Starts with %
      lappend st_tp $tp
    } else {
      lappend st_mt $tp
    }
  }
  set result [closure typed_$name $st_mt $st_tp [make_content content]]\n
  append result [cps !$name $memtypes {} $content(get_type)]
  return $result
}

proc tplist {l} {
  if {[llength $l] == 0} {
    return ""
  } else {
    return "<[join $l ,\ ]>"
  }
}

proc cps {name memtypes tparams l} {
  if {[llength $l] == 0} {
    return ""
  }
  set val [lindex $l 0]
  regsub -all ! $name "" newname
  set newname ${newname}x
  set memtypes_old $memtypes
  set tparams_old $tparams
  if {[regexp {^([^:]*): (.*)$} $val x resultname value]} {
    lappend memtypes $resultname
  } else {
    lappend memtypes K
    lappend tparams Env
  }
  regsub -all @ $val "${newname}[tplist $tparams]([join [string tolower $memtypes] ,\ ])" val
  regsub -all {\^} $val "" val; # Remove markers for non-wrapped args
  if {[regexp {^([^:]*): (.*)$} $val x resultname value]} {
    set closure_content [make_cont $resultname $value]
  } else {
    set closure_content [make_get_type $val]
  }
  set result [closure $name $memtypes_old $tparams_old $closure_content]
  set name $newname
  return "[cps $name $memtypes $tparams [lrange $l 1 end]]\n$result"
}

proc make_cont {param val} {
  return "  template <class $param>
  void operator()(const $param& [string tolower $param]) const {
    $val;
  }
"}

proc make_get_type {val} {
  return "  template <class Env, class K>
  void get_type(const K& k) const {
    $val;
  }
"}

proc extend_abs {env var val} {
  return "typename ${env}::template extend_abs<$var,$val>::type"
}

proc get_type {code env k} {
  return "${code}.template get_type<$env>($k)"
}

proc get_var_type {env var} {
  return "typename ${env}::template get_var_type<$var>::type"
}

proc make_content {cname} {
  upvar $cname content
  set result ""
  foreach k [lsort [array names content]] {
    append result [make_content_$k $content($k)]
  }
  return $result
}

proc make_content_get_type {args} {}
proc make_content_returns {t} {
  return "typedef $t type;\n"
}
proc make_content_run {code} {
  return "template <class Env>
  type run(const Env& env) const {
    $code
  }
  "}

foreach file $argv {
  regsub {.in$} $file {.hpp} outfile
  set data [getfile $file]
  putfile $outfile [subst $data]
}
