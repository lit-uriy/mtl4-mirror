#!/usr/bin/tclsh

set max_tuple_args 10

proc add_cr {t} {
  return "typename boost::add_reference<typename boost::add_const<$t >::type>::type "
}

proc make_specialization {tpnames} {
  global max_tuple_args
  while {[llength $tpnames] < $max_tuple_args} {
    lappend tpnames "detail::unspecified"
  }
  return <[join $tpnames ", "]>
}

proc make_tuple_type_decl {} {
  global max_tuple_args
  set nargs $max_tuple_args
  set tparams ""
  for {set i 0} {$i < $nargs} {incr i} {
    set Arg Arg$i
    lappend tparams "class $Arg = detail::unspecified"
  }
  set tparams [join $tparams ", "]
  puts "  template <$tparams>"
  puts "  struct tuple;"
}

proc make_tuple_type {nargs} {
  global max_tuple_args
  set tparams ""
  for {set i 0} {$i < $nargs} {incr i} {
    set Arg Arg$i
    set arg arg$i
    set tparam "class $Arg"
    lappend tparams $tparam
    lappend targs $Arg
    lappend constargs "[add_cr $Arg] $arg"
    lappend constargnames $arg
  }
  set constargname0 [lindex $constargnames 0]
  set constargnamerest [join [lrange $constargnames 1 end] ", "]
  if {$nargs == 2} {append constargnamerest ", 0"} ; # Disambiguator
  set constargs [join $constargs ", "]
  set constargnames [join $constargnames ", "]
  set tparams [join $tparams ", "]
  set tpnames [join $targs ", "]
  if {$nargs == $max_tuple_args} {
    set specialization ""
  } else {
    set specialization [make_specialization $targs]
  }
  if {$nargs >= 2} {
    set nam1 [expr {$nargs - 1}]
    set part0 "tuple1<[lindex $targs 0]>"
    set part1 "tuple$nam1<[join [lrange $targs 1 end] ,\ ]>"
    puts "  namespace detail {"
    puts "    template <$tparams>"
    puts "    struct tuple$nargs: public append_tuple<$part0, $part1 > {"
    puts "      typedef append_tuple<$part0, $part1 > base;"
    puts ""
    puts "      tuple${nargs}() {}"
    puts "      tuple${nargs}(const tuple$nargs& o): base(o) {}"
    puts ""
    puts "      template <class U>"
    puts "      tuple${nargs}(const U& o): base(o) {}"
    puts "      template <class U>"
    puts "      tuple${nargs}& operator=(const U& o) {base::operator=(o); return *this;}"
    puts ""
    puts "      tuple${nargs}($constargs): base(${part0}($constargname0, 0), ${part1}($constargnamerest)) {}"
    puts "    };"
    puts "  }"
    puts ""
  }
  puts "  template <$tparams>"
  puts "  struct tuple$specialization: public detail::tuple$nargs<$tpnames> {"
  puts "    typedef detail::tuple$nargs<$tpnames> base;"
  puts "    tuple() {}"
  puts "    tuple(const tuple& o): base(o) {}"
  puts ""
  puts "    template <class U>"
  puts "    tuple(const U& o): base(o) {}"
  puts "    template <class U>"
  puts "    tuple& operator=(const U& o) {base::operator=(o); return *this;}"
  puts ""
  puts "    tuple($constargs): base($constargnames) {}"
  if {$nargs == 1} { ; # Special case for disambiguating member construction
    puts ""
    puts "    template <class Mem>"
    puts "    tuple(const Mem& m, int): base(o,0) {}"
  }
  puts "  };"
}

proc make_tuple_type_0 {} {
  global max_tuple_args
  set tparams ""
  set specialization [make_specialization ""]
  puts "  template <>"
  puts "  struct tuple$specialization: public detail::tuple0 {"
  puts "    typedef detail::tuple0 base;"
  puts "    tuple() {}"
  puts "    tuple(const tuple& o): base(o) {}"
  puts ""
  puts "    template <class U>"
  puts "    tuple(const U& o): base(o) {}"
  puts "    template <class U>"
  puts "    tuple& operator=(const U& o) {base::operator=(o); return *this;}"
  puts "  };"
}

proc make_tuple_type_1 {} {
  global max_tuple_args
  set tparams ""
  set specialization [make_specialization "Arg0"]
  puts "  template <class Arg0>"
  puts "  struct tuple$specialization: public detail::tuple1<Arg0> {"
  puts "    typedef detail::tuple1<Arg0> base;"
  puts "    tuple() {}"
  puts "    tuple(const tuple& o): base(o) {}"
  puts ""
  puts "    explicit tuple([add_cr Arg0] a0): base(a0) {}"
  puts "    template <class U>"
  puts "    explicit tuple(const U& o): base(o) {}"
  puts "    template <class U>"
  puts "    tuple& operator=(const U& o) {base::operator=(o); return *this;}"
  puts ""
  puts "    template <class U>" ; # Disambiguator
  puts "    tuple(const U& o, int): base(o, 0) {}"
  puts "  };"
}

proc make_make_tuple {n} {
  set tparams ""; set tupleargs ""; set arguses ""; set args ""
  for {set i 0} {$i < $n} {incr i} {
    set Arg Arg$i
    set arg arg$i
    lappend tparams "class $Arg"
    lappend tupleargs "typename detail::make_tuple_arg<$Arg>::type"
    lappend arguses "typename detail::make_tuple_arg<$Arg>::type($arg)"
    lappend args "const $Arg& $arg"
  }
  set tparams [join $tparams ", "]
  set rettype "tuple<[join $tupleargs ,\ ]>"
  set arguses [join $arguses ", "]
  set args [join $args ", "]
  if {$n == 1} {append arguses ", 0"}
  puts ""
  if {$n != 0} {
    puts "  template <$tparams>"
  }
  puts "  $rettype"
  puts "  make_tuple($args) {"
  puts "    return ${rettype}($arguses);"
  puts "  }"
}

proc make_tie {n} {
  set tparams ""; set tupleargs ""; set arguses ""; set args ""
  for {set i 0} {$i < $n} {incr i} {
    set Arg Arg$i
    set arg arg$i
    lappend tparams "class $Arg"
    lappend tupleargs "$Arg&"
    lappend arguses "$arg"
    lappend args "$Arg& $arg"
  }
  set tparams [join $tparams ", "]
  set rettype "tuple<[join $tupleargs ,\ ]>"
  set arguses [join $arguses ", "]
  set args [join $args ", "]
  if {$n == 1} {append arguses ", 0"}
  puts ""
  if {$n != 0} {
    puts "  template <$tparams>"
  }
  puts "  $rettype"
  puts "  tie($args) {"
  puts "    return ${rettype}($arguses);"
  puts "  }"
}

proc make_tuple_type_general {n} {
  if {$n == 0} {
    make_tuple_type_0
  } elseif {$n == 1} {
    make_tuple_type_1
  } else {
    make_tuple_type $n
  }
  make_make_tuple $n
  make_tie $n
}

puts "#ifndef STD_TUPLE_GENERATED_HPP"
puts "#define STD_TUPLE_GENERATED_HPP"
puts ""
puts "#ifndef STD_TUPLE_IN_LIB"
puts "#error \"Must include <tuple> for this library\""
puts "#endif"
puts ""
puts "#include \"traits.hpp\""
puts "#include \"utility.hpp\""
puts "#include \"tuple0.hpp\""
puts "#include \"tuple1.hpp\""
puts "#include \"append_tuple.hpp\""
puts "#include \"any_holder.hpp\""
puts ""
puts ""
puts "namespace STD_TUPLE_NS {"
puts "  namespace detail {"
puts "    struct unspecified {};"
puts ""
puts "    template <class T>"
puts "    struct make_tuple_arg_2 {"
puts "      typedef typename boost::remove_const<"
puts "                         typename boost::remove_reference<T>::type"
puts "                       >::type type;"
puts "    };"
puts ""
puts "    template <class T>"
puts "    struct make_tuple_arg_2<STD_TUPLE_NS::any_holder<T> > {typedef T type;};"
puts ""
puts "    template <class T>"
puts "    struct make_tuple_arg {"
puts "      typedef typename ct_if<"
puts "                         boost::is_array<T>::value, "
puts "                         [add_cr T], "
puts "                         make_tuple_arg_2<T> >::type::type type;"
puts "    };"
puts "  }"
puts ""
make_tuple_type_decl ; # Put non-specialized version first
for {set i 0} {$i <= $max_tuple_args} {incr i} {
  make_tuple_type_general $i
}
puts "}"
puts ""
puts "#endif // STD_TUPLE_GENERATED_HPP"
