import os
from pathlib import Path

def create_kc_fun_facts_documents():
    """Create test documents with fun facts about Kansas City, Missouri."""
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Document 1: History and Origins
    doc1_content = """
# Kansas City, Missouri: Historical Fun Facts and Origins

## The Name Game

### Why "Kansas" City is in Missouri
Kansas City, Missouri was incorporated in 1850, a full 11 years before Kansas even became a state in 1861. The city was named after the Kansas River (also called the Kaw River), which was itself named after the Kansa Native American tribe. The Missouri city came first, and Kansas City, Kansas wasn't incorporated until 1872, making it the younger sibling despite what many assume.

### The Town of Kansas
The city was originally called the "Town of Kansas" when it was founded in 1838. It was renamed Kansas City in 1889 to distinguish it from the state of Kansas and to sound more metropolitan. Before that, it was also briefly known as the "City of Kansas" from 1853 to 1889.

### Westport's Role
Before Kansas City proper existed, Westport (now a neighborhood) was the main settlement, founded in 1833 by John Calvin McCoy. It served as the final supply point for wagon trains heading out on the Santa Fe, California, and Oregon trails. Westport was an independent city until Kansas City annexed it in 1897.

## Founding Fathers and Early Days

### The Father of Kansas City
John Calvin McCoy is considered the "Father of Kansas City." He not only founded Westport but also laid out the original town of Kansas in 1838. He chose the location specifically because it was the natural rock landing on the Missouri River, perfect for steamboat traffic.

### The Great Border War
During the Civil War era, Kansas City was right on the Kansas-Missouri border during "Bleeding Kansas" - the violent confrontations between pro-slavery and anti-slavery forces. The city changed hands several times during the war and was the site of the Battle of Westport in 1864, sometimes called the "Gettysburg of the West."

### Order No. 11
In 1863, Union General Thomas Ewing issued General Order No. 11, which forced the evacuation of rural areas in four Missouri counties, including Jackson County where Kansas City is located. This devastating order displaced thousands of residents and led to the burning of most rural homes and farms in the area.

## Prohibition Era Peculiarities

### The Wettest City in America
During Prohibition (1920-1933), Kansas City was known as one of the "wettest" cities in the country, meaning alcohol flowed freely despite being illegal. Under the political machine of Tom Pendergast, the city had over 300 speakeasies and jazz clubs operating openly.

### The Pendergast Machine
Tom "Boss Tom" Pendergast ran Kansas City's political machine from 1925 to 1939. His organization was so powerful that he hand-picked Harry S. Truman for the U.S. Senate in 1934. Pendergast's machine controlled everything from garbage collection to police appointments, and his influence made Kansas City a haven for jazz, gambling, and bootlegging.

### Underground Tunnels
Rumor has it that an extensive network of underground tunnels connected various speakeasies and illegal gambling establishments during Prohibition. While some utility tunnels did exist, the extent of the "secret" tunnel system has been greatly mythologized over the years.

## Presidential Connections

### Harry S. Truman
The 33rd President of the United States got his political start in Kansas City. Before becoming president, Truman:
- Worked as a haberdasher (men's clothing store owner) at 104 West 12th Street
- Served as a Jackson County judge (an administrative, not judicial position)
- His store failed in 1922, and he didn't pay off the debts until he became a U.S. Senator
- The Truman Home in Independence (a KC suburb) is now a National Historic Site

### Presidential Visits
Kansas City has hosted every president from Theodore Roosevelt to Joe Biden, except for Franklin Pierce and James Buchanan (who served before the city's prominence). The city was particularly important during the railroad era as a major stop for presidential campaign trains.

## Walt Disney's KC Connection

### Laugh-O-Gram Studios
Walt Disney founded his first animation company, Laugh-O-Gram Studios, in Kansas City in 1921 at 1127 East 31st Street. The company went bankrupt in 1923, prompting Disney to move to Hollywood. The building still stands today and has been renovated as "Thank You, Walt Disney, Inc."

### Mickey Mouse's Conception
Mickey Mouse was reportedly conceived on a train ride from New York back to Kansas City in 1928, after Disney lost the rights to his first character, Oswald the Lucky Rabbit. Disney claimed he was inspired by a pet mouse he had while working in his KC studio.

### Disney's First Employees
Several Kansas City residents became Disney's first animators and followed him to California, including Ub Iwerks (who actually drew the first Mickey Mouse cartoons), Rudolf Ising, and Hugh Harman.

## The Great Flood of 1951

### Devastating Disaster
The flood of July 1951 was the second-worst flood in Kansas City history, causing $250 million in damages (about $2.8 billion in today's dollars). The flood killed 17 people and displaced 200,000 residents.

### Industrial Impact
The flood destroyed the city's stockyards and packing plants in the West Bottoms, accelerating the decline of KC's meatpacking industry. It also wiped out the city's original Municipal Airport, leading to the construction of what would become KCI Airport.

### Engineering Marvel Response
The disaster led to massive flood control projects, including the construction of Tuttle Creek Dam in Kansas and Smithville Lake Dam in Missouri, making Kansas City one of the most flood-protected major cities in America today.

## Unique Historical Quirks

### More Boulevards Than Paris
Kansas City has more boulevards than any city except Paris, France. The city boasts more than 135 miles of boulevards and parkways, thanks to the "City Beautiful" movement and landscape architect George Kessler's 1893 park and boulevard plan.

### The First Shopping Center
Country Club Plaza, opened in 1923, was the first shopping center in the world designed to accommodate shoppers arriving by automobile. It was the first suburban shopping district in the United States and served as a model for shopping centers nationwide.

### Hallmark's Hometown
Joyce Hall founded Hallmark Cards in Kansas City in 1910, starting by selling postcards out of two shoeboxes. The company invented modern gift wrap in 1917 when they ran out of tissue paper at Christmas and substituted decorative envelope lining papers. Hallmark remains headquartered in Kansas City and is still family-owned.

## Civil War Firsts

### First Battle of the Civil War (Arguably)
The Battle of Boonville (June 17, 1861) in Missouri is considered by some historians to be the first organized land action of the Civil War, predating the First Battle of Bull Run by over a month.

### Jesse James Origins
The notorious outlaw Jesse James was from the Kansas City area (born in nearby Kearney, Missouri). His criminal career began as a Confederate guerrilla during the Civil War. His childhood home is now a museum, complete with the original bullet hole from a Pinkerton raid.

## Architectural Innovations

### The First Mall Art Museum
The Nelson-Atkins Museum of Art, opened in 1933, was one of the first museums in the country to be built in a suburban/mall-like setting with extensive grounds, rather than in a dense urban core. Its massive lawn with the famous Shuttlecocks sculpture has become an iconic KC image.

### Union Station
When it opened in 1914, Union Station was the third-largest train station in the country. It could handle 250 trains daily and had the country's first Fred Harvey restaurant. After declining with the railroads, it was beautifully restored and reopened in 1999 as a cultural center.

### The Liberty Memorial
The Liberty Memorial, dedicated in 1926, is the United States' official World War I memorial and museum. At its dedication, it was the only time in history that the five Allied commanders from WWI were together in one place: Belgium's King Albert, Britain's Admiral Beatty, Italy's General Diaz, France's Marshal Foch, and America's General Pershing.
"""

    # Document 2: Food, Culture, and Music
    doc2_content = """
# Kansas City, Missouri: Food, Culture, and Music Fun Facts

## BBQ Capital Claims

### The BBQ Holy Trinity
Kansas City is the only major BBQ city that traditionally serves all types of meat: pork (pulled pork, ribs), beef (brisket, burnt ends), sausage, and even lamb and chicken. Most other BBQ capitals specialize in one or two meats. This makes KC the "melting pot" of American barbecue.

### Burnt Ends Invention
Burnt ends, now a BBQ delicacy worldwide, were invented in Kansas City. They started as free scraps given away at Arthur Bryant's in the 1970s - the crispy, fatty point end of the brisket that was considered too tough to serve. Now they're often the most expensive item on BBQ menus.

### The Sauce Revolution
Kansas City-style BBQ sauce (thick, molasses-based, sweet and tangy) was popularized by Henry Perry in the 1920s. Perry, considered the "Father of Kansas City Barbecue," operated out of an old trolley barn and served BBQ wrapped in newspaper. His sauce recipe influenced all the major KC BBQ restaurants that followed.

### BBQ Restaurant Density
Kansas City has more BBQ restaurants per capita than any other U.S. city - over 100 in the metro area. That's roughly one BBQ joint for every 22,000 residents. Memphis, by comparison, has about one per 35,000 residents.

### Presidential BBQ
LC's Bar-B-Q has served numerous presidents and celebrities. When Barack Obama visited in 2014, he jumped the line (with permission from other customers) and bought $1,400 worth of BBQ for everyone in the restaurant. The staff framed his receipt.

### The Great BBQ Families
Kansas City BBQ is dominated by multi-generational family businesses:
- The Bryant family (Arthur Bryant's, founded 1908)
- The Gates family (Gates BBQ, famous for "Hi, may I help you?" greeting)
- The Boyd family (LC's Bar-B-Q)
- The Edwards family (BB's Lawnside BBQ)

## Jazz Heritage

### The Paris of the Plains
In the 1920s and 1930s, Kansas City was known as the "Paris of the Plains" for its jazz scene. The city had more than 100 nightclubs featuring jazz, and musicians would play all night in cutting contests (competitive jam sessions).

### Charlie "Bird" Parker
Jazz legend Charlie Parker was born in Kansas City, Kansas, but grew up in Kansas City, Missouri. He developed his revolutionary bebop style playing in KC jazz clubs. His childhood home at 1516 Olive Street is marked with a historical plaque, and the city hosts an annual Charlie Parker Jazz Festival.

### Count Basie's Kingdom
Count Basie developed his signature sound at the Reno Club in Kansas City in the 1930s. His band was "discovered" there by producer John Hammond, who heard them on a late-night radio broadcast. The Basie band's rhythm section became the model for all swing bands that followed.

### 18th and Vine
The 18th and Vine District was the heart of African American culture in KC and one of the cradles of jazz. It housed:
- The Mutual Musicians Foundation (still operating as a late-night jam session venue)
- The Gem Theater
- The original Negro Leagues Baseball Museum
- Over 50 jazz clubs within a few blocks

### The Longest Jam Session
Kansas City allegedly hosted one of the longest jam sessions in jazz history. In December 1933, a cutting contest between Coleman Hawkins and Lester Young, with Ben Webster and Herschel Evans, supposedly lasted until past dawn, with some accounts claiming it went on for 12+ hours.

### Mary Lou Williams
Jazz pianist and composer Mary Lou Williams, known as the "First Lady of Jazz," developed her style in Kansas City. She was one of the few female instrumentalists of her era and mentored younger musicians like Dizzy Gillespie and Charlie Parker.

## Unique Food Inventions

### The Happy Meal Predecessor
The Kansas City-based chain Katz Drug Store claims to have invented the kids' meal concept in the 1950s, decades before McDonald's Happy Meal (1979). They called it the "Katz Kitten Club" meal and included a toy with purchase.

### Winstead's and the Steakburger
Winstead's Drive-In claims to have invented the "steakburger" in 1940 - a burger made from ground steak rather than regular ground beef. They also claim to have introduced the concept of the "combo meal" and the paper skirt on burgers to catch drippings.

### The Ice Cream Cone Machine
The waffle cone rolling machine was invented in Kansas City by Carl Taylor in the 1920s. Before this, ice cream cones had to be rolled by hand, limiting production. His invention made mass production of ice cream cones possible.

### Wishbone Salad Dressing
The Wishbone restaurant in Kansas City created the Italian salad dressing that became Wishbone brand dressing in 1945. The restaurant was located in the Wishbone Building (shaped like a wishbone) until it was demolished in the 1980s.

## Cultural Oddities

### The Toy and Miniature Museum
Kansas City has one of the largest collections of fine-scale miniatures and antique toys in the world. The museum houses over 72,000 objects, including the world's largest collection of marbles and a miniature railroad that took 35 years to build.

### Fountain City
Kansas City has more fountains than any city except Rome, with over 200 registered fountains. The city even has an official "City Fountain" position responsible for maintaining them. The tradition started with the Humane Society installing water fountains for horses in the 1890s.

### The Sea Life Aquarium... in the Middle of America
Despite being about 1,000 miles from the nearest ocean, Kansas City has a major aquarium at Crown Center. It features a 260,000-gallon shark tank with a walk-through tunnel. The logistics of maintaining ocean life this far inland involve flying in seawater concentrate weekly.

### Boulevard Brewing
Boulevard Brewing Company, founded in 1989, was the first brewery to open in Kansas City since Prohibition ended. It's now the largest specialty brewer in the Midwest. Their Tank 7 Farmhouse Ale was named after a fermenter tank that supposedly had supernatural properties.

## Sports Food Traditions

### The Tailgating Capital
Arrowhead Stadium's parking lot is considered one of the premier tailgating experiences in all of sports. Fans typically arrive 5-6 hours before kickoff, and the parking lot has its own ZIP code (64129). Some season ticket holders are known more for their tailgate spreads than their seats.

### Stadium Food Innovations
Arrowhead Stadium introduced several stadium food innovations:
- First stadium to have a BBQ restaurant inside (1989)
- Introduced "burnt end nachos" as stadium food
- First NFL stadium with a craft brewery partnership (Boulevard)

### The Z-Man Sandwich
Joe's Kansas City Bar-B-Que (formerly Oklahoma Joe's) created the Z-Man sandwich: beef brisket with provolone cheese and onion rings on a kaiser roll. It's been called the best sandwich in America by numerous food publications and has inspired countless imitators.

## Music Beyond Jazz

### Cowtown Sound
In the 1950s-60s, Kansas City developed the "Cowtown Sound" or "Kansas City Sound" in country music, blending country with jazz and blues influences. Buck O'Neil and his Cowtown Serenaders were pioneers of this style.

### Tech N9ne's Strange Music
Kansas City is home to Strange Music, one of the most successful independent hip-hop labels in the world, founded by rapper Tech N9ne. The label's snake and bat logo is one of the most recognizable in underground hip-hop.

### The Vanguard
The punk/alternative scene in KC produced bands like The Get Up Kids, Coalesce, and Season to Risk in the 1990s, making the city an unlikely emo/hardcore punk hub. The scene centered around venues like The Hurricane and El Torreon Ballroom.

## Culinary Quirks

### The Tenderloin Wars
Kansas City claims to have perfected the pork tenderloin sandwich, with dozens of places serving plate-sized breaded tenderloins. The "war" over who has the best has been ongoing since the 1950s, with Kitty's Cafe and In-A-Tub among the top contenders.

### Cheese Slicer Innovation
The rotating cheese slicer used in delis worldwide was invented in Kansas City by Willis Johnson in 1925. He sold the patent to Kraft, which made it the standard for deli counters across America.

### The Original Food Truck City
Kansas City had "lunch wagons" serving workers in the West Bottoms stockyards as early as the 1890s, arguably making it one of the first cities with food trucks. These wagons served everything from tamales to sandwiches to workers who couldn't leave their posts.

### Russell Stover's Start
Russell Stover Candies was founded in Kansas City in 1923 as "Mrs. Stover's Bungalow Candies." They introduced the first chocolate-covered ice cream bar called "Eskimo Pie" and became America's largest specialty chocolate manufacturer.

## Adult Beverage History

### The Original Sin City
Before Las Vegas earned the title, Kansas City was known as "Sin City" or "The Paris of the Plains" due to its liberal attitudes toward alcohol, gambling, and entertainment during Prohibition and the Pendergast era.

### Cocktail Innovation
The Horsefeather cocktail (whiskey, ginger beer, and bitters) was allegedly invented at the Mutual Musicians Foundation in Kansas City during the 1920s as a way to make bootleg whiskey more palatable.

### Brewery Heritage
Before Prohibition, Kansas City had over 24 breweries. The Imperial Brewing Company building still stands as apartments, and many old brewery caves are now used for storage throughout the city. Some caves maintain a natural 62°F year-round temperature, perfect for aging.
"""

    # Document 3: Modern Kansas City and Quirky Facts
    doc3_content = """
# Kansas City, Missouri: Modern Marvels and Quirky Facts

## Underground City

### SubTropolis
Kansas City is home to SubTropolis, the world's largest underground business complex. This 55,000,000 square foot (1,100 acres) underground space is carved out of limestone and includes:
- Roads wide enough for tractor-trailers
- Rail access for freight
- Over 8 miles of illuminated paved roads
- Storage for the U.S. Postal Service, National Archives, and major corporations
- A constant 65-70°F temperature year-round
- Space that's still growing through active mining

### The Cheese Caves
The USDA stores hundreds of millions of pounds of government cheese in converted limestone caves under Kansas City. At one point, the government stored over 1.4 billion pounds of cheese in these caves, part of dairy price support programs. The caves maintain perfect conditions for cheese aging without any climate control costs.

### Underground Film Storage
Major Hollywood studios store original film reels in KC's underground facilities. The limestone caves provide ideal preservation conditions: consistent temperature, low humidity, and protection from natural disasters. Films from Warner Bros, Universal, and Paramount are all stored here.

### Cave Businesses
Over 50 businesses operate in various KC caves, including:
- Ford Motor Company (parts storage)
- Hallmark Cards (inventory)
- The National Archives (records preservation)
- Various data centers taking advantage of natural cooling
- Wine storage facilities
- Even an underground climbing gym

## Transportation Oddities

### The Airport Far, Far Away
Kansas City International Airport (KCI) is famously far from the city - about 20 miles northwest of downtown. It was built there intentionally in 1972 to avoid noise complaints and allow for expansion. The original design allowed passengers to park and be at their gate within 60 seconds.

### The Three-Terminal Mistake
KCI's original three-terminal design was revolutionary in 1972 but became obsolete after 9/11 security requirements. The circular terminals with gates on both sides made adding modern security nearly impossible. A new single terminal opened in 2023, finally solving the 20-year problem.

### Streetcar Comeback
Kansas City had one of the most extensive streetcar systems in the country until 1957. The modern KC Streetcar, which opened in 2016, is free to ride - one of the few major city transit systems in America with no fare. It's funded by a special taxing district along the route.

### The Paseo Bridge Naming Controversy
The Paseo Bridge was renamed the Martin Luther King Jr. Bridge, then renamed back to Paseo after a controversial vote, creating one of the most complicated naming disputes in American infrastructure history.

## Tech and Innovation

### Google Fiber's First City
In 2011, Kansas City became the first city in the world to get Google Fiber, offering internet speeds of 1 gigabit per second. This sparked a tech boom and earned KC the nickname "Silicon Prairie." Property values in fiberhood areas increased by up to 10%.

### The Startup Village
The area around Google Fiber's initial installation became known as the "Startup Village," where entrepreneurs bought cheap houses and turned them into hacker homes and startup incubators. Some houses sold for as little as $3,000 before the tech boom.

### Cerner's Medical Campus
Cerner Corporation (now Oracle Cerner) built a $4.45 billion "smart campus" in Kansas City - one of the largest corporate campuses in the world. The campus has its own hiking trails, restaurants, and even a museum. It's designed to house 16,000 employees.

### The First Computer Art
The Nelson-Atkins Museum hosted one of the first computer art exhibitions in 1965, featuring works created on an IBM 7094 computer. This was revolutionary at a time when most people had never seen a computer.

## Weird Geography

### Larger Than You Think
Kansas City, Missouri covers 319 square miles, making it the 23rd largest city by area in the United States. It's actually larger than Philadelphia, Boston, and San Francisco combined. This is due to aggressive annexation policies in the 1940s-1960s.

### The State Line Road
State Line Road literally runs along the border between Missouri and Kansas. You can stand with one foot in each state. Businesses on the road have different tax rates depending on which side their front door faces. Some bars used to serve different strength beer based on which side you sat on.

### Four States in a Day
From Kansas City, you can drive to four different states (Missouri, Kansas, Iowa, and Nebraska) in under 2 hours. This makes KC one of the most centrally located major cities in America.

### The Northland
Kansas City has a huge area called "The Northland" north of the Missouri River that's technically part of the city but feels suburban. It contains the airport, hundreds of subdivisions, and was mostly farmland until the 1970s. Many Northlanders rarely go south of the river.

## Sports Peculiarities

### The Loudest Stadium
Arrowhead Stadium holds the Guinness World Record for loudest outdoor stadium at 142.2 decibels (louder than a jet engine at takeoff). The design of the stadium creates a natural acoustic bowl that amplifies crowd noise.

### The Curse of Marty Ball
Before the Chiefs' recent success, they suffered through "Martyball" - the conservative play-calling of coach Marty Schottenheimer (1989-1998). Despite regular season success, the Chiefs lost all three home playoff games under Marty, leading to the phrase "Martyball" becoming synonymous with playoff failure.

### Negro Leagues Baseball Museum
Kansas City is home to the only museum in the world dedicated to Negro Leagues Baseball. The Kansas City Monarchs were the longest-running and most successful Negro League franchise, producing more major league players than any other team, including Jackie Robinson and Satchel Paige.

### The Scout Statue
The iconic Scout statue overlooking the city was originally created for the 1915 Panama-Pacific Exposition in San Francisco. Kansas City citizens loved it so much they raised money to buy it permanently. It's now on the city seal and the name of the local Boy Scout council.

## Modern Cultural Oddities

### The Costume District
Kansas City's West Bottoms transforms into the world's largest year-round costume and vintage district. Over 30 massive warehouses sell costumes, and during October, it becomes a haunted house complex with 20+ haunted attractions - the most concentrated in one area worldwide.

### Apartment Crocodile
In 2019, a Kansas City man was discovered keeping a 7-foot crocodile in his apartment bathtub for 40 years. "Catfish" the crocodile had his own bedroom and was fed chicken and fish. He was relocated to a sanctuary after being discovered.

### The Money Museum
The Federal Reserve Bank of Kansas City has a Money Museum where you can see a real $1 million cube of money, lift a gold bar worth $400,000+, and take home a bag of shredded money (about $165 worth when whole).

### Hair Museum
Leila's Hair Museum in Independence (KC suburb) is the only hair museum in the world, featuring over 3,000 pieces of hair art and jewelry, including hair from celebrities like Marilyn Monroe, Abraham Lincoln, and Queen Victoria.

## Environmental Extremes

### Temperature Swings
Kansas City holds the record for the most extreme temperature change in 24 hours: 85 degrees. On February 11, 1899, the temperature rose from -22°F to 63°F. The city regularly experiences 50+ degree temperature swings in a single day.

### Tornado Alley Adjacent
While not directly in Tornado Alley, Kansas City has been hit by 13 tornadoes since 1950. The 1957 Ruskin Heights tornado (F5) killed 44 people and led to improved tornado warning systems nationwide.

### The Great Ice Storm
The 2002 ice storm left 270,000 KC residents without power for up to two weeks in January. Trees still show damage from the weight of ice. It caused $64 million in damage and changed how the city manages its urban forest.

## Economic Oddities

### More Foreign Trade Zone Space Than Any Inland City
Kansas City has more Foreign Trade Zone space than any other inland U.S. city. This allows companies to import goods without immediately paying customs duties, making KC a major distribution hub.

### Hallmark Christmas Ornament Economy
Hallmark's Keepsake Ornaments, all designed in Kansas City, have created a $500+ million secondary market. Some ornaments from the 1970s sell for thousands of dollars. The annual Ornament Premiere event draws collectors from around the world.

### The Garment District That Was
From 1940-1980, Kansas City was the second-largest garment manufacturing center in America after New York. Brands like Nelly Don employed over 1,000 workers. The massive buildings are now loft apartments, but you can still see the freight elevators and factory floors.

## Celebrity Connections

### Famous KC Natives You Didn't Know
- Robert Altman (director)
- Jean Harlow (actress, original "Blonde Bombshell")
- Ginger Rogers (actress/dancer)
- Casey Kasem (radio host)
- Paul Rudd (actor, huge Chiefs fan)
- Jason Sudeikis (actor/comedian)
- Tech N9ne (rapper)
- Janelle Monáe (singer/actress)
- Eric Stonestreet (Modern Family actor)

### The Queer Eye Effect
When Netflix's "Queer Eye" filmed Season 3 and 4 in Kansas City (2018-2019), it led to a measurable increase in tourism called the "Queer Eye Effect." Restaurants featured on the show reported 30-50% increases in business.

## Architectural Quirks

### The Upside-Down Building
The Bartle Hall Convention Center has pylons that look like upside-down ice cream cones or shuttlecocks. They're actually supposed to represent the campfires of Native American tribes. Locals call them "the bow ties" or "the lipstick tubes."

### The Western Auto Sign
The Western Auto sign atop the former company building is Kansas City's most iconic sign. Though the company went out of business in 2003, the sign is maintained by a condo association and lit for special occasions, visible for miles.

### Jesse James's Bank
The Clay County Savings Association building in nearby Liberty, Missouri (KC metro) was the site of the first successful daylight bank robbery in the United States (February 13, 1866), committed by Jesse James and his gang.
"""

    # Document 4: Kansas City Superlatives and Records
    doc4_content = """
# Kansas City, Missouri: Superlatives, Records, and Extreme Facts

## World Records and Firsts

### Medical Milestones
- **First Successful Insulin Treatment in the U.S.** (1922): Dr. Gilman C. Lowrey at Kansas City General Hospital administered the first successful insulin treatment west of the Mississippi River, just months after its discovery in Canada.
- **First Hospital-Based Emergency Medical Service**: Kansas City General Hospital created the first hospital-based ambulance service in 1900, revolutionizing emergency medical care.
- **Largest Medical Center Between Chicago and Denver**: The University of Kansas Medical Center (technically in Kansas City, Kansas, but part of the metro) is the region's largest medical facility.

### Retail Innovations
- **First Shopping Cart Patent**: The folding shopping cart was perfected and patented in Kansas City by Arthur Kosted in 1946, improving on earlier designs.
- **First Price Tag**: The Emery, Bird, Thayer Department Store in Kansas City introduced the first fixed price tags in the Midwest (1870s), ending the tradition of haggling.
- **Largest Costume Jewelry Collection**: The Toy and Miniature Museum houses over 20,000 pieces of costume jewelry, the world's largest public collection.

### Engineering Marvels
- **First Concrete Street in America**: A block of concrete pavement at 27th and Prospect (1909) was the first concrete street in the United States, and parts of it still exist.
- **Longest Continuously Operating Farmers Market**: City Market has operated continuously since 1857, making it one of the longest-running public markets in the Midwest.
- **First Major Bridge with a Moveable Span West of the Mississippi**: The original Hannibal Bridge (1869) was the first bridge to span the Missouri River and revolutionized westward expansion.

## Size and Scale Records

### The Biggest Everything
- **World's Largest Shuttlecock Sculptures**: The four 18-foot tall shuttlecocks at the Nelson-Atkins Museum are the world's largest, weighing 5,500 pounds each.
- **Largest Column-Free Ballroom**: The Bartle Hall ballroom spans 388 feet without columns, one of the largest column-free spaces in the world.
- **Largest WWI Memorial**: The Liberty Memorial is the United States' largest and most comprehensive World War I memorial.
- **Largest Miniature Railroad**: At Union Station, the Model Railroad Experience features 8,000 square feet of model trains, one of the largest permanent model railroad displays in the country.

### Underground Superlatives
- **World's Largest Underground Business Complex**: SubTropolis at 55 million square feet
- **Largest Underground Storage Facility**: Hunt Midwest's underground facilities store everything from stamps to coffee
- **Most Underground Roadway**: Over 8 miles of underground paved, lighted roads suitable for tractor-trailers
- **Deepest Business Complex**: Some areas of SubTropolis are 160 feet below ground

## Economic and Business Records

### Corporate Giants Born in KC
- **H&R Block**: Started in Kansas City in 1955, now the world's largest tax preparation company
- **Russell Stover**: Largest American chocolate manufacturer by volume
- **Hallmark**: Largest greeting card manufacturer in the world, producing 10,000 different designs annually
- **AMC Theatres**: Started in KC in 1920, now the world's largest movie theater chain
- **Hostess Brands**: Twinkies were invented in Kansas City (though this is disputed with Chicago)

### Trading and Commerce
- **Second Largest Railroad Hub**: In the early 1900s, only Chicago moved more rail freight than Kansas City
- **Hard Winter Wheat Trading Capital**: The Kansas City Board of Trade sets the global price for hard red winter wheat
- **Most Railroad Track**: Kansas City has more miles of railroad track than any other U.S. city except Chicago
- **Largest Inland Foreign Trade Zone**: 10,000+ acres of Foreign Trade Zone space

## Weather and Climate Extremes

### Temperature Records
- **Hottest Temperature**: 113°F (July 14, 1954)
- **Coldest Temperature**: -23°F (December 22 and 23, 1989)
- **Fastest Temperature Rise**: 75°F in 4 hours (February 11, 1899)
- **Most 100°F Days in a Year**: 53 days in 1936
- **Latest 90°F Day**: November 3 (1893)
- **Earliest 90°F Day**: March 21 (1907)

### Precipitation Records
- **Wettest Year**: 62.31 inches (1961)
- **Driest Year**: 18.32 inches (1953)
- **Most Rain in 24 Hours**: 8.71 inches (September 12-13, 1961)
- **Deepest Snow**: 25 inches (March 23, 1912)
- **Most Snow in One Season**: 67.0 inches (1911-1912)

### Storm Records
- **Deadliest Tornado**: The 1957 Ruskin Heights F5 tornado killed 44 people
- **Costliest Hailstorm**: April 10, 2001 caused $2 billion in damage with softball-sized hail
- **Longest Power Outage**: 2002 ice storm left some without power for 16 days
- **Highest Wind Gust**: 98 mph (June 22, 1969)

## Cultural and Entertainment Records

### Festival and Event Superlatives
- **Longest-Running Renaissance Festival**: Kansas City Renaissance Festival (since 1977) is one of the longest continuously running in the U.S.
- **Largest St. Patrick's Day Parade West of the Mississippi**: KC's parade dates to 1873
- **Oldest Blues and Jazz Festival**: The Kansas City Blues and Jazz Festival has run continuously since 1980
- **Largest Ethnic Festival in the Midwest**: The Ethnic Enrichment Festival features 60+ cultures

### Sports Records
- **Loudest Outdoor Stadium**: Arrowhead Stadium at 142.2 decibels
- **Longest Playoff Drought Ended**: Royals ended a 29-year playoff drought in 2014 (longest in major American sports at the time)
- **Most Comeback Wins in NFL Playoffs**: Chiefs have 9 playoff comeback wins under Patrick Mahomes
- **First MLB Team with Artificial Turf**: The Royals at Municipal Stadium (1970)

### Media Milestones
- **First FM Radio Station West of the Mississippi**: KOZY-FM (1948)
- **Oldest Continuously Published African American Newspaper West of the Mississippi**: The Kansas City Call (since 1919)
- **First Television Station in Kansas City**: WDAF-TV (1949), still broadcasting today

## Demographic and Social Records

### Population Extremes
- **Fastest Growth Period**: 1860-1870 grew 351% (from 4,418 to 32,260)
- **Peak Population**: 507,087 in 1970
- **Most Millionaires Per Capita (1900)**: Had more millionaires per capita than any U.S. city
- **Largest Suburban Annexation**: Annexed 225 square miles between 1940-1970

### Diversity Records
- **Most Refugee Resettlement in Missouri**: Kansas City resettles more refugees than any other Missouri city
- **Largest Sudanese Population in the Midwest**: Over 10,000 Sudanese immigrants
- **First U.S. City to Elect Openly Gay Black Mayor Pro Tem**: Quinton Lucas's appointment in 2017
- **Most Languages Spoken in Public Schools**: KC Public Schools serve students speaking 60+ languages

## Unique "Only in KC" Facts

### Things That Only Exist Here
- **Only WWI Museum Designated by Congress as National Memorial**: Liberty Memorial
- **Only Museum Dedicated to Negro Leagues Baseball**: Negro Leagues Baseball Museum
- **Only Museum Dedicated to Miniatures in the Central U.S.**: National Museum of Toys and Miniatures
- **Only Hair Art Museum in the World**: Leila's Hair Museum
- **Only Steamboat Museum in the Midwest**: Arabia Steamboat Museum

### Unusual Distinctions
- **Farthest Inland City with Direct Ocean Shipping**: Via Missouri River barge to Mississippi to Gulf
- **Most Boulevard Miles Designed by One Architect**: George Kessler designed 130+ miles
- **Most French Impressionist Paintings Between Chicago and West Coast**: Nelson-Atkins Museum
- **Largest Collection of Native American Art in Missouri**: Nelson-Atkins Museum

## Food and Beverage Records

### Consumption Records
- **Most BBQ Sauce Consumed Per Capita**: Kansas Citians consume 5.8 bottles of BBQ sauce per person annually (national average is 1.3)
- **Most Fountains Per Capita After Rome**: Over 200 public fountains
- **First City to Fluoridate Water Supply**: Added fluoride to water in 1946
- **Most Breweries Before Prohibition in Missouri**: 24 breweries operating in 1910

### Restaurant Records
- **Oldest Continuously Operating BBQ Restaurant**: LC's Bar-B-Q (since 1908, under various names)
- **First Drive-Through Restaurant Window**: In-A-Tub (1946) claims first drive-through window
- **Largest Restaurant Seating**: Stroud's Oak Ridge Manor could seat 1,200 people (now closed)
- **Most James Beard Award Nominations Per Capita (2015-2020)**: For cities under 1 million population

## Transportation and Infrastructure Records

### Bridge and Road Facts
- **Most Bridges in Missouri**: Over 300 bridges in city limits
- **Longest Continuously Used Bridge**: Broadway Bridge since 1956
- **First City to Use Salt on Winter Roads**: Started in 1914
- **Most Highway Loops**: Three complete highway loops (I-435, I-635, and inner loop)

### Airport Records
- **Fastest Gate-to-Curb Design**: Original KCI allowed 60-second curb-to-gate
- **Most Airlines Served at Peak**: 32 airlines in 1978
- **Largest Single Terminal Project**: New $1.5 billion terminal (2023) biggest infrastructure project in city history

## Educational and Research Records

### Academic Achievements
- **Most Urban Universities**: 15 colleges and universities in metro area
- **Oldest Continuously Operating Law School West of Mississippi**: UMKC School of Law (1895)
- **Largest Community College in Missouri**: Metropolitan Community College
- **First Dental School West of Mississippi**: Kansas City Dental College (1881)

### Research Milestones
- **First Use of Lithotripsy in Midwest**: Saint Luke's Hospital (1985)
- **Largest Animal Health Corridor**: KC Animal Health Corridor has highest concentration of animal health companies in world
- **Most Patents Per Capita in Missouri**: Averages 150+ patents per 100,000 residents annually

## Environmental and Nature Records

### Green Space Achievements
- **First City Beautiful Plan West of Mississippi**: 1893 Kessler Plan
- **Most Parkland Per Capita for Major Cities**: 13.5% of city is parkland
- **Largest Municipal Rose Garden in Midwest**: Laura Conyers Smith Rose Garden (4,000 roses)
- **Most Trees Planted in Single Year**: 10,000 trees planted in 2020

### Wildlife Records
- **Northernmost Breeding Ground for Scissor-tailed Flycatcher**: Swope Park
- **Largest Urban Deer Population in Missouri**: Estimated 500+ deer in city limits
- **Most Bird Species Recorded in Missouri Urban Area**: 320+ species documented

## Modern Technology Records

### Digital Achievements
- **First Gigabit City**: Google Fiber launch in 2011
- **Fastest Municipal WiFi**: Free streetcar WiFi at 1GB speed
- **Most Tech Startups Per Capita in Midwest (2015-2020)**: "Silicon Prairie" boom
- **First Smart City Partnership with Cisco**: 2016 smart city initiative

### Innovation Metrics
- **Most Maker Spaces Per Capita in Midwest**: 12 public maker spaces
- **Largest Hackathon in Region**: Hack KC draws 1,000+ participants
- **First City to Use AI for Pothole Detection**: 2019 pilot program
- **Most Electric Vehicle Charging Stations in Missouri**: 500+ public charging points

## Final Superlatives

### The Ultimate KC Facts
- **Most Fountains Except Rome**: 200+ fountains
- **More Boulevards Than Paris**: 135+ miles of boulevards
- **More BBQ Restaurants Per Capita Than Any City**: 100+ BBQ joints
- **Larger Than Boston, San Francisco, and Miami Combined**: 319 square miles
- **More Federal Reserve Currency Printed Than Any Facility Except D.C.**: $40+ billion annually
- **Jazz Capital of the World (1920s-1940s)**: Over 100 jazz clubs at peak
- **Most Extreme Temperature Swing**: 85°F in 24 hours
- **World's Largest Educated Workforce Corridor**: Animal Health Corridor
- **America's Most Centrally Located Major City**: Geographic center of continental U.S.
- **Most Recession-Proof Major City**: Weathered 2008 recession better than any major metro
"""

    # Write the documents
    with open(data_dir / "kc_history_origins.txt", "w", encoding="utf-8") as f:
        f.write(doc1_content)
    
    with open(data_dir / "kc_food_culture_music.txt", "w", encoding="utf-8") as f:
        f.write(doc2_content)
    
    with open(data_dir / "kc_modern_quirky.txt", "w", encoding="utf-8") as f:
        f.write(doc3_content)
    
    with open(data_dir / "kc_superlatives_records.txt", "w", encoding="utf-8") as f:
        f.write(doc4_content)
    
    print(" Kansas City fun facts documents created successfully!")
    print(f"📁 Documents location: {data_dir.absolute()}")
    print("📄 Created files:")
    print("   1. kc_history_origins.txt")
    print("   2. kc_food_culture_music.txt")
    print("   3. kc_modern_quirky.txt")
    print("   4. kc_superlatives_records.txt")

if __name__ == "__main__":
    create_kc_fun_facts_documents()