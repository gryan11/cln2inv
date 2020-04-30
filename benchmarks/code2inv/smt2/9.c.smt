(set-logic LIA)

( declare-const x Int )
( declare-const x! Int )
( declare-const x0 Int )
( declare-const x0! Int )
( declare-const y Int )
( declare-const y! Int )
( declare-const y0 Int )
( declare-const y0! Int )
( declare-const tmp Int )
( declare-const tmp! Int )

( declare-const x_0 Int )
( declare-const x_1 Int )
( declare-const x_2 Int )
( declare-const x0_0 Int )
( declare-const y_0 Int )
( declare-const y_1 Int )
( declare-const y_2 Int )
( declare-const y0_0 Int )

( define-fun inv-f( ( x Int )( x0 Int )( y Int )( y0 Int )( tmp Int ) ) Bool
SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop
)

( define-fun pre-f ( ( x Int )( x0 Int )( y Int )( y0 Int )( tmp Int )( x_0 Int )( x_1 Int )( x_2 Int )( x0_0 Int )( y_0 Int )( y_1 Int )( y_2 Int )( y0_0 Int ) ) Bool
	( and
		( = x x_0 )
		( = x0 x0_0 )
		( = y y_0 )
		( = y0 y0_0 )
		( = x0_0 x_0 )
		( = y0_0 y_0 )
		( >= x_0 0 )
		( <= x_0 2 )
		( <= y_0 2 )
		( >= y_0 0 )
	)
)

( define-fun trans-f ( ( x Int )( x0 Int )( y Int )( y0 Int )( tmp Int )( x! Int )( x0! Int )( y! Int )( y0! Int )( tmp! Int )( x_0 Int )( x_1 Int )( x_2 Int )( x0_0 Int )( y_0 Int )( y_1 Int )( y_2 Int )( y0_0 Int ) ) Bool
	( or
		( and
			( = x_1 x )
			( = y_1 y )
			( = x_1 x! )
			( = y_1 y! )
			( = x x! )
			( = x0 x0! )
			( = y y! )
			( = y0 y0! )
			(= tmp tmp! )
		)
		( and
			( = x_1 x )
			( = y_1 y )
			( = x_2 ( + x_1 2 ) )
			( = y_2 ( + y_1 2 ) )
			( = x_2 x! )
			( = y_2 y! )
			(= x0 x0_0 )
			(= x0! x0_0 )
			(= y0 y0_0 )
			(= y0! y0_0 )
			(= tmp tmp! )
		)
	)
)

( define-fun post-f ( ( x Int )( x0 Int )( y Int )( y0 Int )( tmp Int )( x_0 Int )( x_1 Int )( x_2 Int )( x0_0 Int )( y_0 Int )( y_1 Int )( y_2 Int )( y0_0 Int ) ) Bool
	( or
		( not
			( and
				( = x x_1)
				( = x0 x0_0)
				( = y y_1)
				( = y0 y0_0)
			)
		)
		( not
			( and
				( = x_1 4 )
				( not ( not ( = y_1 0 ) ) )
			)
		)
	)
)
SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop
( assert ( not
	( =>
		( pre-f x x0 y y0 tmp x_0 x_1 x_2 x0_0 y_0 y_1 y_2 y0_0  )
		( inv-f x x0 y y0 tmp )
	)
))

SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop
( assert ( not
	( =>
		( and
			( inv-f x x0 y y0 tmp )
			( trans-f x x0 y y0 tmp x! x0! y! y0! tmp! x_0 x_1 x_2 x0_0 y_0 y_1 y_2 y0_0 )
		)
		( inv-f x! x0! y! y0! tmp! )
	)
))

SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop
( assert ( not
	( =>
		( inv-f x x0 y y0 tmp  )
		( post-f x x0 y y0 tmp x_0 x_1 x_2 x0_0 y_0 y_1 y_2 y0_0 )
	)
))

