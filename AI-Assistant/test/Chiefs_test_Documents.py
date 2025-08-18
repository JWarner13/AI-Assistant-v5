import os
from pathlib import Path

def create_chiefs_test_documents():
    """Create test documents with Kansas City Chiefs facts from 2019 season onwards."""
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Document 1: 2019 Season - First Mahomes Super Bowl
    doc1_content = """
# Kansas City Chiefs: 2019 Season - The Comeback Kings

## Season Overview
The 2019 Kansas City Chiefs season marked the franchise's 50th in the National Football League and their 60th overall. Under head coach Andy Reid's seventh season, the Chiefs finished 12-4 in the regular season, earning the #2 seed in the AFC playoffs.

## Regular Season Highlights

### Patrick Mahomes' MVP-Caliber Season
Despite missing two games due to a dislocated kneecap against Denver, Patrick Mahomes threw for 4,031 yards and 26 touchdowns with only 5 interceptions in 14 games. His passer rating of 105.3 demonstrated elite efficiency even while dealing with injury.

### Offensive Prowess
- The Chiefs averaged 28.2 points per game (5th in NFL)
- Total offense: 379.2 yards per game (6th in NFL)
- Travis Kelce: 97 receptions for 1,229 yards
- Tyreek Hill: 58 receptions for 860 yards despite injury
- Damien Williams emerged as lead back with 498 rushing yards

### Defensive Improvements
The defense showed significant improvement in the second half of the season:
- Allowed just 11.5 points per game over final 6 regular season games
- Frank Clark recorded 8 sacks in first season with Chiefs
- Tyrann Mathieu added veteran leadership with 75 tackles and 4 interceptions
- Chris Jones dominated with 9 sacks from the interior

## Playoff Run - Three Historic Comebacks

### Wild Card Round: Texans 31, Chiefs 51
- Trailed 24-0 in the second quarter
- Scored 41 unanswered points
- Mahomes: 321 yards, 5 TDs
- First team in NFL history to win playoff game by 20+ after trailing by 20+

### Divisional Round: Chiefs 51, Texans 31
The Chiefs hosted Houston and delivered a historic comeback:
- Down 24-0 early in the second quarter
- Scored 28 points in the second quarter alone
- Patrick Mahomes threw for 321 yards and 5 touchdowns
- Travis Kelce: 10 catches, 134 yards, 3 TDs

### AFC Championship: Chiefs 35, Titans 24
- Trailed 17-7 and 17-14
- Mahomes rushed for 27-yard touchdown
- 28 unanswered points to close the game
- Earned first Super Bowl appearance in 50 years

## Super Bowl LIV Victory

### Chiefs 31, 49ers 20 (February 2, 2020)
**The Comeback:**
- Trailed 20-10 with 8:53 remaining
- Scored 21 unanswered points in final 6:13
- Mahomes: 286 yards, 2 passing TDs, 1 rushing TD
- Named Super Bowl MVP at age 24

**Key Moments:**
- 3rd and 15 conversion to Tyreek Hill for 44 yards
- Damien Williams' go-ahead touchdown with 2:44 left
- Williams' 38-yard touchdown run to seal victory
- Ended 50-year championship drought

### Andy Reid's Redemption
- First Super Bowl victory as head coach
- 222nd career win
- Cemented legacy as one of greatest coaches in NFL history

## Season Awards and Honors
- Patrick Mahomes: Super Bowl MVP
- Frank Clark: Pro Bowl selection
- Tyreek Hill: All-Pro Second Team
- Travis Kelce: First-Team All-Pro
- Chris Jones: Pro Bowl selection
- Tyrann Mathieu: First-Team All-Pro

## Franchise Records Set
- First Super Bowl championship since Super Bowl IV (1970)
- Largest comeback in franchise playoff history (24 points)
- Patrick Mahomes youngest Super Bowl MVP since Ben Roethlisberger

## Cultural Impact
The victory parade on February 5, 2020, drew an estimated 800,000 fans to downtown Kansas City, with schools closing and the city painted red. The phrase "Run it Back" immediately became the rallying cry for 2020.
"""

    # Document 2: 2020 Season - The Super Bowl Defense
    doc2_content = """
# Kansas City Chiefs: 2020 Season - Defending Champions

## Season Overview
The 2020 season saw the Chiefs navigate the COVID-19 pandemic while defending their Super Bowl title. They finished 14-2 in the regular season, earning the #1 seed in the AFC for the first time since 2018.

## Regular Season Dominance

### Historic Start
- Won first 6 games of the season
- Only loss before Week 17 came to Las Vegas Raiders
- Rested starters in Week 17 loss to Chargers
- Best regular season record in franchise history (14-2)

### Patrick Mahomes' Excellence
**Regular Season Stats:**
- 4,740 passing yards (2nd in NFL)
- 38 passing touchdowns (2nd in NFL)
- 6 interceptions
- 108.2 passer rating
- 316 rushing yards and 2 rushing TDs

### Offensive Explosion
**Team Statistics:**
- Led NFL with 29.6 points per game
- 415.8 total yards per game (1st in NFL)
- 303.4 passing yards per game (1st in NFL)

**Key Performers:**
- Travis Kelce: 105 catches, 1,416 yards, 11 TDs (NFL record for TE yards)
- Tyreek Hill: 87 catches, 1,276 yards, 15 TDs
- Clyde Edwards-Helaire: 803 rushing yards as rookie
- Sammy Watkins: 421 yards, 2 TDs

### Defensive Statistics
- 22.6 points allowed per game (16th in NFL)
- Chris Jones: 7.5 sacks
- Frank Clark: 6 sacks
- L'Jarius Sneed emerged as shutdown corner
- Tyrann Mathieu: 6 interceptions

## Notable Regular Season Games

### Week 11: Chiefs 35, Raiders 31 (Sunday Night Football)
- Mahomes to Kelce for game-winning TD with 28 seconds left
- Overcame 14-point deficit
- Mahomes: 348 yards, 2 TDs

### Week 12: Chiefs 27, Buccaneers 24
- Victory over Tom Brady and future Super Bowl champions
- Tyreek Hill: 269 total yards, 3 TDs
- First quarter: Hill had 203 yards and 2 TDs

### Week 14: Chiefs 33, Dolphins 27
- Mahomes: 393 yards, 2 passing TDs, 1 rushing TD
- Comeback from 10-0 deficit
- Mecole Hardman 67-yard punt return TD

## Playoff Run to Super Bowl LV

### Divisional Round: Chiefs 22, Browns 17
- Chad Henne relief appearance after Mahomes concussion
- Henne's crucial 3rd down scramble to seal game
- Defense held Browns to 17 points

### AFC Championship: Chiefs 38, Bills 24
- Hosted AFC Championship for 3rd straight year
- Mahomes: 325 yards, 3 TDs
- Dominated second half 21-3
- Advanced to back-to-back Super Bowls

## Super Bowl LV Disappointment

### Buccaneers 31, Chiefs 9 (February 7, 2021)
**Struggles:**
- Offensive line decimated by injuries
- Started two backup tackles
- Mahomes pressured on 29 of 56 dropbacks (most in Super Bowl history)
- No touchdowns scored (field goals only)
- Mahomes: 270 yards, 0 TDs, 2 INTs

**Key Issues:**
- Eric Fisher and Mitchell Schwartz injuries
- Penalties: 11 for 120 yards
- Couldn't establish running game
- Tom Brady's 7th Super Bowl victory

## Season Achievements

### Individual Honors
- Patrick Mahomes: NFL Offensive Player of the Year nominee
- Travis Kelce: First-Team All-Pro, Pro Bowl
- Tyreek Hill: First-Team All-Pro, Pro Bowl
- Chris Jones: Pro Bowl selection
- Tyrann Mathieu: Pro Bowl selection

### Team Records
- Best regular season winning percentage in franchise history (.875)
- First back-to-back Super Bowl appearances in franchise history
- Travis Kelce set NFL record for receiving yards by tight end
- 10 consecutive wins to start season (including playoffs from 2019)

## COVID-19 Impact
- Limited attendance at Arrowhead (22% capacity)
- Multiple schedule adjustments
- Enhanced safety protocols
- Virtual meetings and altered practice schedules

## Legacy Notes
Despite the Super Bowl loss, the 2020 Chiefs established themselves as the NFL's newest dynasty, with three straight AFC Championship Game appearances and back-to-back Super Bowl appearances. The offensive line injuries in the Super Bowl would lead to a complete overhaul in the 2021 offseason.
"""

    # Document 3: 2021-2022 Seasons - The Dynasty Continues
    doc3_content = """
# Kansas City Chiefs: 2021-2022 Seasons - Sustained Excellence

## 2021 Season: Redemption and Heartbreak

### Regular Season (12-5)
The Chiefs retooled their offensive line after Super Bowl LV, bringing in Joe Thuney, Orlando Brown Jr., Creed Humphrey, and Trey Smith. The investment paid immediate dividends.

### Offensive Resurgence
**Patrick Mahomes Stats:**
- 4,839 passing yards (led NFL)
- 37 passing touchdowns
- 13 interceptions
- 98.5 passer rating
- 381 rushing yards, 2 TDs

**Receiving Corps:**
- Tyreek Hill: 111 catches, 1,239 yards, 9 TDs
- Travis Kelce: 92 catches, 1,125 yards, 9 TDs
- Mecole Hardman: 59 catches, 693 yards, 2 TDs
- Byron Pringle: 42 catches, 568 yards, 5 TDs

### Midseason Struggles
- Started 3-4, worst start of Mahomes era
- Defense ranked last in NFL through Week 8
- Turned season around with 9-1 finish
- Won AFC West for 6th consecutive year

### Playoff Run - The 13 Seconds Game

**Wild Card: Chiefs 42, Steelers 21**
- Mahomes: 404 yards, 5 TDs
- Travis Kelce: 5 catches, 108 yards, 1 TD
- Complete domination from start to finish

**Divisional Round: Chiefs 42, Bills 36 (OT)**
"The Greatest Playoff Game Ever"
- 13 seconds: Chiefs scored to force OT after Bills took lead
- Mahomes to Kelce and Hill for 44 yards in 10 seconds
- Harrison Butker 49-yard field goal to force OT
- Mahomes: 378 yards, 3 passing TDs, 1 rushing TD
- Overtime TD pass to Kelce for victory

**AFC Championship: Bengals 27, Chiefs 24 (OT)**
- Led 21-3 before Bengals comeback
- First AFC Championship Game loss at home under Reid/Mahomes
- Ended bid for third straight Super Bowl appearance

## 2022 Season: Return to Glory

### Regular Season (14-3)
Despite trading Tyreek Hill to Miami, the Chiefs adapted with a more balanced offensive approach.

### Offensive Evolution Without Hill
**Patrick Mahomes NFL MVP Season:**
- 5,250 passing yards (led NFL)
- 41 passing touchdowns (led NFL)
- 12 interceptions
- 105.2 passer rating
- First 5,000-yard season

**New Receiving Distribution:**
- Travis Kelce: 110 catches, 1,338 yards, 12 TDs
- JuJu Smith-Schuster: 78 catches, 933 yards, 3 TDs
- Marquez Valdes-Scantling: 42 catches, 687 yards, 2 TDs
- Rookie Skyy Moore: 22 catches, 250 yards
- Kadarius Toney (midseason trade): 14 catches, 171 yards, 2 TDs

### Defensive Improvements
- Chris Jones: 15.5 sacks (career high)
- George Karlaftis (rookie): 6 sacks
- Nick Bolton: 180 tackles, 2 interceptions
- L'Jarius Sneed: 108 tackles, 3 INTs

### Notable 2022 Regular Season Moments

**Week 11: Overtime Loss in Germany**
- Chiefs 9, Buccaneers 12 (first loss)
- International Series game in Munich
- Struggled without injured receivers

**Week 13: Chiefs 34, Bengals 31**
- Revenge game for AFC Championship
- Mahomes: 336 yards, 2 TDs
- Game-winning field goal as time expired

**Week 15: Chiefs 30, Texans 24 (OT)**
- Mahomes to Jerick McKinnon walk-off TD
- 502 total yards of offense
- 10th straight AFC West title-clinching victory

## Super Bowl LVII Victory

### The Road Through Playoffs

**Wild Card: BYE (1st seed)**

**Divisional: Chiefs 27, Jaguars 20**
- Patrick Mahomes ankle injury in first quarter
- Gutted out victory on one leg
- Chad Henne crucial minutes in relief

**AFC Championship: Chiefs 23, Bengals 20**
- Revenge for previous year's loss
- Harrison Butker game-winning 45-yard field goal
- Mahomes: 326 yards, 2 TDs despite ankle injury
- Chris Jones crucial late-game pressure

### Super Bowl LVII: Chiefs 38, Eagles 35
**February 12, 2023 - State Farm Stadium, Glendale, AZ**

**The Classic:**
- Highest-scoring Super Bowl in history (73 points)
- Patrick Mahomes: 182 yards, 3 TDs on injured ankle
- Aggravated ankle injury in 2nd quarter, returned to lead comeback

**Key Moments:**
- James Bradberry holding penalty with 1:54 left
- Harrison Butker go-ahead field goal with 8 seconds left
- Mahomes Super Bowl MVP (3 TDs despite injury)
- Kadarius Toney 65-yard punt return (longest in SB history)
- Skyy Moore fumble recovery for touchdown

**Historical Significance:**
- Second Super Bowl in four years
- Mahomes joins elite company with 2 Super Bowl MVPs
- Andy Reid's second championship
- First team since Patriots (2003-04) to win Super Bowl after losing previous year

## 2022 Season Honors

### Individual Achievements
- Patrick Mahomes: NFL MVP, Offensive Player of the Year, Super Bowl MVP
- Travis Kelce: First-Team All-Pro (4th time)
- Chris Jones: Second-Team All-Pro
- Patrick Mahomes: First player with 5,000 yards and 40 TDs since 2013
- Creed Humphrey: Second-Team All-Pro

### Franchise Milestones
- 14-3 best record since 2020
- 7th straight AFC West title
- 5th straight AFC Championship Game
- 3rd Super Bowl appearance in 4 years
- Patrick Mahomes playoff record: 11-3

## Dynasty Status Confirmed
With two Super Bowl victories in four years and five consecutive AFC Championship Game appearances, the Chiefs established themselves as the NFL's current dynasty, with Patrick Mahomes as the undisputed best quarterback in football.
"""

    # Document 4: 2023-2024 Seasons - The Back-to-Back Quest
    doc4_content = """
# Kansas City Chiefs: 2023-2024 Seasons - Championship DNA

## 2023 Season: The Receiving Corps Crisis

### Regular Season Overview (11-6)
The 2023 season tested the Chiefs like never before, with the NFL's worst receiving corps by drops and separation metrics, yet they still found ways to win.

### Offensive Struggles and Adaptation

**Patrick Mahomes' "Down" Year:**
- 4,183 passing yards
- 27 passing touchdowns (career low as starter)
- 14 interceptions
- 92.6 passer rating (lowest as starter)
- Still led team to division title

**Receiving Issues:**
- NFL-leading 28 dropped passes
- Kadarius Toney regression and drops
- Skyy Moore minimal impact
- Marquez Valdes-Scantling inconsistency
- Travis Kelce: 93 catches, 984 yards, 5 TDs (lowest since 2015)

**Bright Spots:**
- Rashee Rice emergence: 79 catches, 938 yards, 7 TDs as rookie
- Isiah Pacheco: 935 rushing yards, 7 TDs
- Hollywood Brown signed for 2024 (injured in preseason)

### Defensive Excellence
The defense carried the team through offensive struggles:
- Chris Jones: 10.5 sacks, 29 QB hits
- George Karlaftis: 10.5 sacks (breakout year)
- L'Jarius Sneed: All-Pro caliber season
- Trent McDuffie: Emerged as elite corner
- Nick Bolton: 145 tackles
- Allowed 17.3 points per game (2nd in NFL)

### Notable 2023 Regular Season Games

**Week 14: Chiefs 21, Bills 24**
- Kadarius Toney offside negated go-ahead TD
- Controversial ending
- Travis Kelce's lateral attempt failed

**Week 15: Chiefs 19, Patriots 27**
- Worst home loss of Mahomes era
- Multiple turnovers and drops
- Raised serious concerns about offense

**Christmas Day: Chiefs 14, Raiders 20**
- Lost at home on Christmas
- Two defensive touchdowns for Raiders
- Lowest point of regular season

### 2023 Playoff Run - Defense Wins Championships

**Wild Card: Chiefs 26, Dolphins 7**
- Freezing conditions at Arrowhead (-4°F at kickoff)
- Mahomes: 262 yards, 1 TD
- Defense dominated Miami's offense

**Divisional: Chiefs 27, Bills 24**
- Tyler Bass missed field goal with 1:47 left
- Mahomes to Kelce connection in clutch
- Sweet revenge for Week 14 loss

**AFC Championship: Chiefs 17, Ravens 10**
- Defense shut down NFL MVP Lamar Jackson
- Travis Kelce: 11 catches, 116 yards, 1 TD
- Lowest-scoring AFC Championship in years

### Super Bowl LVIII: The Overtime Thriller

**Chiefs 25, 49ers 22 (OT)**
**February 11, 2024 - Allegiant Stadium, Las Vegas**

**Historic Victory:**
- First Super Bowl to go to overtime under new playoff rules
- Patrick Mahomes: 333 yards, 2 passing TDs, 1 rushing TD
- Game-winning TD pass to Mecole Hardman in OT
- Mahomes' 3rd Super Bowl MVP

**Key Moments:**
- Harrison Butker's 57-yard field goal (longest in SB history)
- Mahomes scramble for first down on 4th and 1
- Leo Chenal blocked extra point
- Mike Danna forced fumble in OT
- "Tom Brady-like" dynasty moment

**Back-to-Back Championships:**
- First repeat champions since Patriots (2003-04)
- Cemented dynasty status
- Patrick Mahomes: 3-0 in Super Bowls
- Andy Reid joins elite coaches with 3+ titles

## 2024 Season: The Three-Peat Quest (Ongoing)

### Current Season Progress (as of late 2024)
The Chiefs are pursuing an unprecedented third consecutive Super Bowl, something never achieved in the Super Bowl era.

### Roster Improvements
**Offensive Additions:**
- Xavier Worthy (1st round rookie): Speed threat
- Hollywood Brown: Veteran receiver (when healthy)
- DeAndre Hopkins (midseason trade): Veteran presence
- Kareem Hunt return: RB depth after Pacheco injury

**Key Departures:**
- L'Jarius Sneed traded to Titans
- Mike Edwards released
- Willie Gay Jr. left in free agency

### 2024 Season Highlights (Through Week 17)
**Record: 15-1** (best in franchise history)
- Clinched #1 seed in AFC
- 9th consecutive AFC West title
- Won 14 straight one-score games dating to 2023

**Patrick Mahomes Resurgence:**
- Over 3,900 passing yards
- 26 passing touchdowns
- 11 interceptions
- Multiple game-winning drives
- "Houdini" moments in crucial situations

**Offensive Weapons:**
- Travis Kelce: Rejuvenated with Hopkins arrival
- Xavier Worthy: 50+ catches, 600+ yards, 6 TDs
- DeAndre Hopkins: Immediate impact after trade
- Rashee Rice: Strong start before injury
- Kareem Hunt: 700+ yards after October signing

### Notable 2024 Games

**Week 2: Chiefs 26, Bengals 25**
- Harrison Butker game-winning field goal
- Fourth straight win over Cincinnati
- Mahomes: 2 TDs despite early struggles

**Week 9: Chiefs 30, Buccaneers 24 (OT)**
- Monday Night Football overtime thriller
- Kareem Hunt walk-off touchdown
- DeAndre Hopkins debut: 8 catches, 86 yards

**Week 11: Chiefs 21, Bills 30**
- Only loss of regular season
- Josh Allen's statement game
- Ended 15-game winning streak

**Week 13: Black Friday Special**
- Chiefs 19, Raiders 17
- Special Black Friday Amazon game
- 15th straight one-score victory

### Historical Context and Dynasty Status

**Dynasty Achievements (2019-2024):**
- 3 Super Bowl victories (LIV, LVII, LVIII)
- 4 Super Bowl appearances in 6 years
- 6 consecutive AFC Championship Games
- 9 straight AFC West titles
- 76-20 regular season record (.792)
- 15-3 playoff record

**Patrick Mahomes Legacy:**
- 3x Super Bowl MVP
- 2x NFL MVP (2018, 2022)
- 6x Pro Bowl
- 3x All-Pro
- Already considered top-10 QB all-time
- On pace for GOAT conversation

**Andy Reid Coaching Tree:**
- 3 Super Bowl wins with Chiefs
- 4th all-time in wins
- Revolutionized offensive football
- Developed multiple successful coordinators

## The Three-Peat Quest

As the 2024 playoffs approach, the Chiefs stand on the precipice of history. No team has ever won three consecutive Super Bowls. The combination of Mahomes' excellence, Reid's coaching, and a defense that rises in big moments has created the NFL's most dominant dynasty since the Brady-Belichick Patriots.

**Keys to Three-Peat:**
- Mahomes' clutch gene
- Improved receiving corps with Hopkins
- Elite defense led by Chris Jones
- Championship experience
- Best record in AFC securing home-field advantage

The Kansas City Chiefs have transformed from a franchise with a 50-year championship drought to the NFL's gold standard, with Patrick Mahomes establishing himself as the face of the league and potentially the greatest quarterback of his generation.
"""

    # Write the documents
    with open(data_dir / "chiefs_2019_first_championship.txt", "w", encoding="utf-8") as f:
        f.write(doc1_content)
    
    with open(data_dir / "chiefs_2020_defending_champions.txt", "w", encoding="utf-8") as f:
        f.write(doc2_content)
    
    with open(data_dir / "chiefs_2021_2022_dynasty_builds.txt", "w", encoding="utf-8") as f:
        f.write(doc3_content)
    
    with open(data_dir / "chiefs_2023_2024_back_to_back.txt", "w", encoding="utf-8") as f:
        f.write(doc4_content)
    
    print("🏈 Kansas City Chiefs test documents created successfully!")
    print(f"📁 Documents location: {data_dir.absolute()}")
    print("📄 Created files:")
    print("   1. chiefs_2019_first_championship.txt (2019 Season - First Super Bowl)")
    print("   2. chiefs_2020_defending_champions.txt (2020 Season - Back-to-back attempt)")
    print("   3. chiefs_2021_2022_dynasty_builds.txt (2021-2022 Seasons - Return to glory)")
    print("   4. chiefs_2023_2024_back_to_back.txt (2023-2024 Seasons - Dynasty cemented)")
    print("\n🏆 Documents cover the Patrick Mahomes era from 2019-2024!")
    print("⚡ Includes Super Bowl wins, playoff runs, records, and key moments!")

if __name__ == "__main__":
    create_chiefs_test_documents()