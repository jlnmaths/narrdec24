from otree.api import *
import numpy as np

doc = """
Your app description
"""


class C(BaseConstants):
    NAME_IN_URL = 'mental_models'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 3


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass

def likertScale(label, low, high, n, blank = False):
    if low != '':
        CHOICES = [(0, str(0) + " (" + low + ")")]
    else:
        CHOICES = [(0, str(0))]
    for x in range(1, n):
        CHOICES = CHOICES + [(x, str(x))]
    if high != '':
        CHOICES = CHOICES + [(n, str(n) + " (" + high + ")")]
    else:
        CHOICES = CHOICES + [(n, str(n))]
    return models.IntegerField(
        choices=CHOICES,
        label=label,
        widget=widgets.RadioSelectHorizontal,
        blank=blank,
    )

class Player(BasePlayer):
    assessment = models.IntegerField(min=0, max=100, label="What do you think is the probability that the main observation (Y) behind the question mark (?) is a 1?")
    score = models.FloatField()
    certainty =likertScale(
        'On a scale of 0 to 10, on which 0 means "very uncertain" and 10 means "very certain", how certain are you that your assessment was correct?',
        '', '', 10)
    certainty_sen = models.StringField(label="Would you have preferred not to send a message?",
        choices=['Yes', 'No'])
    prob = models.FloatField()
    starttime = models.IntegerField(initial=0)
    treatment = models.IntegerField(initial=0)
    finishtime = models.IntegerField(initial=0)
    true_y = models.IntegerField(initial=2)
    payround = models.IntegerField(initial=0)
    startexptime = models.IntegerField(initial=0)
    datademandtime = models.IntegerField(initial=0) #the moment the button to reveal is clicked (captcha then appears)
    hintdemandtime = models.IntegerField(initial=0) #the moment the button to reveal is clicked (captcha then appears)
    signaldemandtime = models.IntegerField(initial=0)  # the moment the button to reveal is clicked (captcha then appears)
    signaltime = models.IntegerField(initial=0)  # the moment the data is revealed (after solving captcha)
    datatime = models.IntegerField(initial=0)  #the moment the data is revealed (after solving captcha)
    hinttime = models.IntegerField(initial=0)  #the moment the hint is revealed (after solving captcha)
    startquiztime = models.IntegerField(initial=0)
    startquiz2time = models.IntegerField(initial=0)
    startquiz3time = models.IntegerField(initial=0)
    n_selection = models.StringField(
        choices=['Hint 1 for the selected table', 'Hint 2 for the selected table'],
        widget=widgets.RadioSelect,
        label= "Which message do you want to send to the receiver?")

    t_selection = models.StringField(
        choices=['Table 1', 'Table 2'],
        widget=widgets.RadioSelect,
        label= "Which table do you want to send to the receiver?"
    )
    prolific_id = models.StringField(default=str(" "))
    table = models.StringField()
    row_order = models.StringField()
    column_order = models.StringField()
    det_stoch_order = models.StringField()
    state_order = models.StringField()
    ord = models.StringField()
    sord = models.StringField()
    ord_2 = models.StringField()
    isdisqualified = models.IntegerField(initial = 0)
    hinttype = models.IntegerField(initial = -1)
    hint_constr = models.LongStringField(label='Can you explain to us in one to two sentences, which characteristics of the data table you paid attention to in order to determine the probability and how you interpreted these?', blank=True)
    quiz1 = models.StringField(label='True or false? The relationship between the main observation (Y) and the side observations is different from row to row.',
        choices=['True', 'False'])
    quiz2 = models.StringField(label="True or false? The events in the 7th row have definitely happened after the events in the first row.",
        choices=['True', 'False'])
    quiz3 = models.StringField(label="True or false? The messages point out a pattern in the visible part of the dataset.",
        choices=['True', 'False'])
    quiz4 = models.StringField(label="True or false? You have to assess the probability that a 1 or a 0 is hidden behind the question mark (?).",
        choices=['True', 'False'])
    quiz5 = models.StringField(label="True or false? With a probability of 50%, you receive a payoff of $5 if there is a 1 hidden behind the question mark (?).",
                               choices=['True', 'False'])
    quiz6 = models.StringField(label="True or false? If the signal is blue, this is evidence indicating that the number behind the question mark (?) is a 1.",
                               choices=['True', 'False'])
    quiz7 = models.StringField(label="True or false? The messages necessarily provide correct information about how the number behind the question mark (?) was generated.",
        choices=['True', 'False'])
# functions
def creating_session(subsession: Subsession):
    import itertools
    import random
    import numpy as np
    treatments = itertools.cycle([2,2,4,5,5,6,7])
    for player in subsession.get_players():
        player.participant.treatment = next(treatments)
        player.treatment = player.participant.treatment
        if subsession.round_number == 1:
            if player.treatment % 4 < 2:
                player.participant.payround = random.randint(1,2)
            if player.treatment % 4 >= 2:
                player.participant.payround = random.randint(1,3)
            player.participant.true_y = random.randint(0,1)
            present = np.random.permutation(3)
            player.participant.order = str(present[0]) + str(present[1]) + str(present[2])
            present = np.random.permutation(3)
            player.participant.order_2 = str(present[0]) + str(present[1]) + str(present[2])
        player.true_y = player.participant.true_y
        player.payround = player.participant.payround
        player.ord = player.participant.order
        player.ord_2 = player.participant.order_2
def set_payoff(player: Player):
    if player.subsession.round_number == player.payround:
        if int(player.true_y) == 1:
            prob = 1- (1 - player.assessment/100)*(1 - player.assessment/100)
        else:
            prob = 1 - (player.assessment/100)*(player.assessment/100)
        player.prob = prob
        pay = float((2.5*np.random.choice([0,1], 1, p=[1-prob, prob])[0])) +2.5 #assessment based payment
        if player.treatment > 3:
            if np.random.choice([0,1], 1, p=[0.5,0.5])[0] == 0:
                pay = float(2.5*player.participant.true_y) +2.5 #bonus based payment
        player.payoff = pay



# PAGES
class Instructions(Page):
    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        player.prolific_id = player.participant.label

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startexptime = int(time.time())
        return dict()


class Instructions_rec_H(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.treatment % 4 < 2

class Instr_rec_ss(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.treatment % 4 > 1

class Instructions_payoff_A(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.treatment < 4

class Instructions_payoff_B(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.treatment > 3

class Instr_betw_H(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 2 and player.treatment % 4 < 2

class Stage1_H(Page):
    form_model = 'player'
    form_fields = ['assessment']

    @staticmethod
    def vars_for_template(player: Player):
        import time
        o_v = np.random.permutation([4, 5, 6, 7, 8, 9, 10, 11])
        o_h = np.random.permutation([0, 1, 2, 3])
        o = np.append(o_h, o_v)  # reordering of columns
        o = np.append(o, np.array([12]))


        present = [int(player.ord[0]), int(player.ord[1]), int(player.ord[2])]     #order of these arrays: d_balanced, d_pro, d_con, s_balanced, s_pro, s_con
        present_2 = [int(player.ord_2[0]), int(player.ord_2[1]), int(player.ord_2[2])]


        # order of these arrays: d_balanced, d_pro, d_con, s_balanced, s_pro, s_con
        hb = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2])
        nb_1 = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1]])  # plus
        nb_2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1]])  # minus
        nb_3 = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]])  # join
        hb = hb[o]  # hb is always the same
        y=0
        plus = nb_1[y][o]
        minus = nb_2[y][o]
        join = nb_3[y][o]

        if y == 0:
            player.table = 'det_balanced'
        if y == 1:
            player.table = 'det_plus'
        if y == 2:
            player.table = 'det_minus'

        # randomize column order

        nb = [plus, minus, join]
        rand = np.random.permutation([0, 1, 2])
        c_1 = nb[rand[0]]
        c_2 = nb[rand[1]]
        c_3 = nb[rand[2]]
        plus = np.where(rand == 0)[0][0] + 1
        join = np.where(rand == 2)[0][0] + 1

        player.starttime = int(time.time())
        player.row_order = str(o)
        player.column_order = str(np.array(['plus', 'minus', 'join'])[rand])  # 0: plus, 1: minus, 2: join


        if player.treatment == 1 or player.treatment == 5:
            keyword = 'The following always holds: If E'+str(plus)+' equals 1, Y also equals 1.'
        if player.treatment == 0 or player.treatment == 4:
            keyword = 'Y is 1 in exactly half of the rows.'


        return dict(
            hb=hb,
            nb_1=c_1,
            nb_2=c_2,
            nb_3=c_3,
            round=player.round_number,
            plus=plus,
            join=join,
            column_order=player.column_order,
            table=player.table,
            row_order=player.row_order,
            keyword=keyword
        )


    @staticmethod
    def live_method(player: Player, data):
        import time
        if int(data) == 1:
            player.datatime = int(time.time())
        if int(data) == 2:
            player.hinttime = int(time.time())
        if int(data) == 3:
            player.datademandtime = int(time.time())
        if int(data) == 4:
            player.hintdemandtime = int(time.time())
        # can just do more cases here!

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        import time
        set_payoff(player)
        player.finishtime = int(time.time())

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1 and player.treatment % 4 < 2


class Stage2_H(Page):
    form_model = 'player'
    form_fields = ['assessment']

    @staticmethod
    def vars_for_template(player: Player):
        import time
        o_v = np.random.permutation([4, 5, 6, 7, 8, 9, 10, 11])
        o_h = np.random.permutation([0, 1, 2, 3])
        o = np.append(o_h, o_v)  # reordering of columns
        o = np.append(o, np.array([12]))

        present = [int(player.ord[0]), int(player.ord[1]), int(player.ord[2])]
        present_2 = [int(player.ord_2[0]), int(player.ord_2[1]), int(player.ord_2[2])]

        # order of these arrays: d_balanced, d_pro, d_con, s_balanced, s_pro, s_con
        hb = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2])
        nb_1 = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1]])  # plus
        nb_2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1]])  # minus
        nb_3 = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]])  # join
        hb = hb[o]  # hb is always the same
        y=0
        plus = nb_1[y][o]
        minus = nb_2[y][o]
        join = nb_3[y][o]

        # randomize column order

        nb = [plus, minus, join]
        rand = np.random.permutation([0, 1, 2])
        c_1 = nb[rand[0]]
        c_2 = nb[rand[1]]
        c_3 = nb[rand[2]]
        plusloc = np.where(rand == 0)[0][0] + 1
        minusloc = np.where(rand == 1)[0][0] + 1

        if y == 0:
            player.table = 'det_balanced'
        if y == 1:
            player.table = 'det_plus'
        if y == 2:
            player.table = 'det_minus'

        if player.treatment == 1 or player.treatment == 5:
            exkeyword = 'The following always holds: If E'+str(plusloc)+' equals 1, Y also equals 1.'
            keyword = 'Y is 1 exactly in exactly half of the rows.'
        if player.treatment == 0 or player.treatment == 4:
            exkeyword = 'Y is 1 exactly in exactly half of the rows.'
            keyword = 'The following always holds: If E'+str(plusloc)+' equals 1, Y also equals 1.'



        player.starttime = int(time.time())
        player.row_order = str(o)
        player.column_order = str(np.array(['plus', 'minus', 'join'])[rand])  # 0: plus, 1: minus, 2: join

        return dict(
            hb=hb,
            nb_1=c_1,
            nb_2=c_2,
            nb_3=c_3,
            round=player.round_number,
            join=join,
            column_order=player.column_order,
            table=player.table,
            row_order=player.row_order,
            keyword=keyword,
            exkeyword=exkeyword,

        )

    @staticmethod
    def live_method(player: Player, data):
        import time
        if int(data) == 1:
            player.datatime = int(time.time())
        if int(data) == 2:
            player.hinttime = int(time.time())
        if int(data) == 3:
            player.datademandtime = int(time.time())
        if int(data) == 4:
            player.hintdemandtime = int(time.time())
        # can just do more cases here!

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        import time
        set_payoff(player)
        player.finishtime = int(time.time())

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 2 and player.treatment % 4 < 2


class Stage3_SS(Page):
    form_model = 'player'
    form_fields = ['assessment']

    @staticmethod
    def vars_for_template(player: Player):
        import time
        import random
        import numpy as np
        o_v = np.random.permutation([4, 5, 6, 7, 8, 9, 10, 11])
        o_h = np.random.permutation([0, 1, 2, 3])
        o = np.append(o_h, o_v)  # reordering of columns
        o = np.append(o, np.array([12]))

        if player.round_number == player.payround:
            ty = player.participant.true_y
        else:
            ty = random.choice([0,1])

        if (player.participant.treatment %4 == 3):
            acc = 90
        if (player.participant.treatment % 4 == 2):
            acc = 60

        if ty == 0:
            p = [acc / 100, 1 - acc / 100]
        else:
            p = [1 - acc / 100, acc / 100]
        signal = np.random.choice(['red (number more likely a 0)', 'blue (number more likely a 1)'], 1, p)[0]


        present = [int(player.ord[0]), int(player.ord[1]), int(player.ord[2])]
        present_2 = [int(player.ord_2[0]), int(player.ord_2[1]), int(player.ord_2[2])]

        # order of these arrays: d_balanced, d_pro, d_con, s_balanced, s_pro, s_con
        hb = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2])
        nb_1 = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                         [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]])  # plus
        nb_2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1]])  # minus
        nb_3 = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]])  # join
        hb = hb[o]  # hb is always the same
        x = player.round_number - 1
        if (x < 3):
            y = present[x]
        else:
            if (x > 2 and x<6):
                y = present_2[x-3]
            else:
                y = present_2[x-6]

        plus = nb_1[y][o]
        minus = nb_2[y][o]
        join = nb_3[y][o]


        if y == 0:
            player.table = 'det_balanced'
        if y == 1:
            player.table = 'det_plus'
        if y == 2:
            player.table = 'det_minus'

        # randomize column order

        nb = [plus, minus, join]
        rand = np.random.permutation([0, 1, 2])
        c_1 = nb[rand[0]]
        c_2 = nb[rand[1]]
        c_3 = nb[rand[2]]
        plusloc = np.where(rand == 0)[0][0] + 1

        keyword_2 = 'I checked to see if any rows had all 4 columns featuring a 1. None of them did, so I selected 0% in every case, as I was sure that it could not be a 1 behind the ?. '#pro 1
        keyword_3 = 'I paid attention to the relationship between the main observation (Y) and the side observation E'+str(plusloc)+'. Specifically, I noticed that whenever E'+str(plusloc)+' was 1, the main observation (Y) was also 1. This consistent pattern suggested a potential effect of E1 on the main observation.'#pro 1
        keyword_4 = 'Nothing stood out to me in any of the tables that made me certain there would be a 1 under the question mark.  So, I figured it was a coin flip in each table.'#pro 50

        player.starttime = int(time.time())
        player.row_order = str(o)
        player.column_order = str(np.array(['plus', 'minus', 'join'])[rand])  # 0: plus, 1: minus, 2: join

        return dict(
            hb=hb,
            nb_1=c_1,
            nb_2=c_2,
            nb_3=c_3,
            round=player.round_number,
            plus=plusloc,
            join=join,
            column_order=player.column_order,
            table=player.table,
            signal=signal,
            acc=acc,
            row_order=player.row_order,
            keyword_2=keyword_2,
            keyword_3=keyword_3,
            keyword_4=keyword_4
        )

    @staticmethod
    def live_method(player: Player, data):
        import time
        if int(data) == 1:
            player.signaltime = int(time.time())
        if int(data) == 22:
            player.hinttime = int(time.time())
            player.hinttype = 0
        if int(data) == 23:
            player.hinttime = int(time.time())
            player.hinttype = 100
        if int(data) == 24:
            player.hinttime = int(time.time())
            player.hinttype = 50
        if int(data) == 3:
            player.signaldemandtime = int(time.time())
        if int(data) == 4:
            player.hintdemandtime = int(time.time())

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        import time
        set_payoff(player)
        player.finishtime = int(time.time())

    @staticmethod
    def is_displayed(player: Player):
        return (player.participant.treatment % 4 > 1)


class Stage3_H(Page):
    form_model = 'player'
    form_fields = ['assessment']

    @staticmethod
    def vars_for_template(player: Player):
        import time
        import random
        import numpy as np
        o_v = np.random.permutation([4, 5, 6, 7, 8, 9, 10, 11])
        o_h = np.random.permutation([0, 1, 2, 3])
        o = np.append(o_h, o_v)  # reordering of columns
        o = np.append(o, np.array([12]))

        if player.round_number == player.payround:
            ty = player.participant.true_y
        else:
            ty = random.choice([0,1])

        if (player.participant.treatment % 4 == 3):
            acc = 90
        if (player.participant.treatment % 4 == 2):
            acc = 60

        if ty == 0:
            p = [acc / 100, 1 - acc / 100]
        else:
            p = [1 - acc / 100, acc / 100]
        signal = np.random.choice(['red', 'blue'], 1, p)[0]

        present = [int(player.ord[0]), int(player.ord[1]), int(player.ord[2])]
        present_2 = [int(player.ord_2[0]), int(player.ord_2[1]), int(player.ord_2[2])]

        # order of these arrays: d_balanced, d_pro, d_con, s_balanced, s_pro, s_con
        hb = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2])
        nb_1 = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                         [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]])  # plus
        nb_2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1]])  # minus
        nb_3 = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]])  # join
        hb = hb[o]  # hb is always the same
        x = player.round_number - 1
        if (x < 3):
            y = present[x]
        else:
            if (x > 2 and x<6):
                y = present_2[x-3]
            else:
                y = present_2[x-6]

        plus = nb_1[y][o]
        minus = nb_2[y][o]
        join = nb_3[y][o]

        # randomize column order

        nb = [plus, minus, join]
        rand = np.random.permutation([0, 1, 2])
        c_1 = nb[rand[0]]
        c_2 = nb[rand[1]]
        c_3 = nb[rand[2]]
        plusloc = np.where(rand == 0)[0][0] + 1
        joinloc = np.where(rand == 2)[0][0] + 1

        if y == 0:
            player.table = 'det_balanced'
        if y == 1:
            player.table = 'det_plus'
        if y == 2:
            player.table = 'det_minus'

        if player.round_number == 7:
            keyword = 'None of the rows has ones for all side observations except the last one.'
        if player.round_number == 8:
            keyword = 'Side observation E'+str(joinloc)+' is always 1 and hence completely uninformative.'
        if player.round_number == 9:
            keyword = 'Side observation E'+str(joinloc)+' is always 1 and hence completely uninformative.'


        player.starttime = int(time.time())
        player.row_order = str(o)
        player.column_order = str(np.array(['plus', 'minus', 'join'])[rand])  # 0: plus, 1: minus, 2: join

        return dict(
            hb=hb,
            nb_1=c_1,
            nb_2=c_2,
            nb_3=c_3,
            round=player.round_number,
            plus=plusloc,
            join=join,
            column_order=player.column_order,
            table=player.table,
            signal = signal,
            acc = acc,
            row_order=player.row_order,
            signalcss=signal,
            keyword=keyword,
        )

    @staticmethod
    def live_method(player: Player, data):
        import time
        if int(data) == 1:
            player.signaltime = int(time.time())
        if int(data) == 2:
            player.hinttime = int(time.time())
        if int(data) == 3:
            player.signaldemandtime = int(time.time())
        if int(data) == 4:
            player.hintdemandtime = int(time.time())
        # can just do more cases here!

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        import time
        set_payoff(player)
        player.finishtime = int(time.time())

    @staticmethod
    def is_displayed(player: Player):
        return player.participant.treatment % 4 >= 2



class Certainty(Page):
    form_model = 'player'
    form_fields = ['certainty']
    def is_displayed(player: Player):
        if player.treatment % 4 < 2:
            return player.round_number <= 2
        else:
            return player.round_number <= 3



class Quiz_A_H(Page):
    form_model = 'player'
    form_fields = ['quiz1', 'quiz2', 'quiz3', 'quiz4', 'quiz5']

    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quiz1='False', quiz2='False', quiz3='True', quiz4='True', quiz5='False')
        if values != solutions:
            return "One or more responses were unfortunately wrong."

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and (player.treatment < 2)

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startquiztime = int(time.time())
        return dict()

class Quiz_B_H(Page):
    form_model = 'player'
    form_fields = ['quiz1', 'quiz2', 'quiz3', 'quiz4', 'quiz5']

    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quiz1='False', quiz2='False', quiz3='True', quiz4='True', quiz5='True')
        if values != solutions:
            return "One or more responses were unfortunately wrong."

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and (player.treatment >= 4) and player.treatment < 6

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startquiztime = int(time.time())
        return dict()

class Hint_Constr(Page):
    form_model = 'player'
    form_fields = ['hint_constr']
    def is_displayed(player: Player):
        if player.treatment % 4 < 2:
            return player.round_number == 2
        else:
            return player.round_number == 3

class Quiz_B_S(Page):
    form_model = 'player'
    form_fields = ['quiz1', 'quiz2', 'quiz3', 'quiz4', 'quiz5', 'quiz6', 'quiz7']

    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quiz1='False', quiz2='False', quiz3='True', quiz4='True', quiz5='True', quiz6='True', quiz7='False')
        if values != solutions:
            return "One or more responses were unfortunately wrong."

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startquiztime = int(time.time())
        if (player.participant.treatment % 4 == 3):
            acc = 9
            compl = 1
            image_path = "9.PNG"
        else:
            acc = 6
            compl = 4
            image_path = "6.PNG"

        return dict(acc=acc,
                    compl=compl,
                    image_path=image_path)

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.treatment > 5


class Quiz_A_S(Page):
    form_model = 'player'
    form_fields = ['quiz1', 'quiz2', 'quiz3', 'quiz4', 'quiz5', 'quiz6', 'quiz7']

    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quiz1='False', quiz2='False', quiz3='True', quiz4='True', quiz5='False', quiz6='True', quiz7='False')
        if values != solutions:
            return "One or more responses were unfortunately wrong."

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startquiztime = int(time.time())

        if (player.participant.treatment % 4 == 3):
            acc = 9
            compl = 1
            image_path = "9.PNG"
        else:
            acc = 6
            compl = 4
            image_path = "6.PNG"

        return dict(acc=acc,
                    compl=compl,
                    image_path=image_path)

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.treatment > 1 and player.treatment < 4

class Instr_rec_S(Page):
    @staticmethod
    def vars_for_template(player: Player):
        if (player.participant.treatment % 4 == 3):
            acc = 9
            compl = 1
            image_path = "9.PNG"
        else:
            acc = 6
            compl = 4
            image_path = "6.PNG"

        return dict(acc=acc,
                    compl=compl,
                    image_path = image_path)

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.treatment % 4 > 1

class Captcha(Page):
    @staticmethod
    def live_method(player: Player, data):
        import time
        if int(data) == 10:
            player.isdisqualified = 1
            player.payoff = 0

    def is_displayed(player: Player):
        return player.subsession.round_number == 1

class ResultsWaitPage(WaitPage):
    pass

class Disqual(Page):
    def is_displayed(player: Player):
        return player.isdisqualified


class Results(Page):
    pass


page_sequence = [Instructions, Captcha, Disqual, Instructions_rec_H, Instr_rec_S, Instructions_payoff_A, Instructions_payoff_B, Quiz_A_H, Quiz_B_H, Quiz_A_S, Quiz_B_S, Stage1_H, Instr_betw_H, Stage2_H, Stage3_SS, Certainty, Hint_Constr]
