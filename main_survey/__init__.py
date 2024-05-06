from otree.api import *
import numpy as np

doc = """
Your app description
"""


class C(BaseConstants):
    NAME_IN_URL = 'mental_models'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 12


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
    assessment = models.IntegerField(min=0, max=100, label="What do you think is the probability that the main observation behind the question mark (?) is a 1?")
    score = models.FloatField()
    certainty =likertScale(
        'On a scale of 0 to 10, on which 0 means "very uncertain" and 10 means "very certain", how certain are you that your assessment was correct?',
        '', '', 10)
    certainty_sen = models.StringField(label="Would you have preferred not to send a hint?",
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
    datatime = models.IntegerField(initial=0)  #the moment the data is revealed (after solving captcha)
    hinttime = models.IntegerField(initial=0)  #the moment the hint is revealed (after solving captcha)
    startquiztime = models.IntegerField(initial=0)
    startquiz2time = models.IntegerField(initial=0)
    n_selection = models.StringField(
        choices=['Hint 1 under the selected table', 'Hint 2 under the selected table'],
        widget=widgets.RadioSelect,
        label= "Which hint do you want to send to the receiver?"
    )
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
    hint_constr = models.LongStringField(label='Can you explain to us in one to two sentences, which characteristics of the data table you paid attention to in order to determine the probability and how you interpreted these?', blank=True)
    quiz1 = models.StringField(label='True or false? The relationship between the main observation and the side observations is different from row to row.',
        choices=['True', 'False'])
    quiz2 = models.StringField(label="True or false? The events in the 7th row have definitely happened after the events in the first row.",
        choices=['True', 'False'])
    quiz3 = models.StringField(label="True or false? The hints can be true, but do not have to be.",
        choices=['True', 'False'])
    quiz4 = models.StringField(label="True or false? You have to assess the probability that a 1 or a 0 is hidden behind the question mark (?).",
        choices=['True', 'False'])
    quiz5 = models.StringField(label="True or false? With a probability of 50%, you receive a payoff of $5 if there is a 1 hidden behind the question mark (?).",
                               choices=['True', 'False'])
    quiz6 = models.StringField(label="True or false? You have to decide between additional data and a hint.",
                               choices=['True', 'False'])
    quizs1 = models.StringField(label='True or false? The relationship between the main observation and explanatory variable is different from row to row.',
        choices=['True ', 'False'])
    quizs2 = models.StringField(label="True or false? The events in the 7th row have definitely happened after the events in the first row.",
        choices=['True ', 'False'])
    quizs3 = models.StringField(label="True or false? The hints can be true, but do not have to be.",
        choices=['True ', 'False'])
    quizs4 = models.StringField(label="You have to choose a hint.",
        choices=['True ', 'False'])
    quizs5 = models.StringField(label="You want to convince the receiver that behind the question mark (?)",
        choices=['...there is a 0.', '...there is a 1.'])
    quizs6 = models.StringField(label="True or false? 50% of the time, Receivers get paid $5 if there is a 1 hidden behind the question mark (?), independent of their assessment.",
                               choices=['True ', 'False'])

# functions
def creating_session(subsession: Subsession):
    import itertools
    import random
    treatments = itertools.cycle([0, 1, 2, 3, 4, 5]) #6,7]) #NNH, NHH, NH, BNH, BHH, BH, SenNB, SenB
    for player in subsession.get_players():
        player.participant.treatment = next(treatments)
        player.treatment = player.participant.treatment
        if subsession.round_number == 1:
            player.participant.payround = random.randint(1,12)
            player.participant.true_y = random.randint(0,1)
            present = np.random.permutation(6)
            player.participant.order = str(present[0]) + str(present[1]) + str(present[2]) + str(present[3]) + str(present[4]) + str(present[5])
            present = np.random.permutation(6)
            presents = np.random.permutation(3)
            player.participant.sorder = str(presents[0]) + str(presents[1]) + str(presents[2])
            player.participant.order_2 = str(present[0]) + str(present[1]) + str(present[2]) + str(present[3]) + str(present[4]) + str(present[5])
        player.true_y = player.participant.true_y
        player.payround = player.participant.payround
        player.ord = player.participant.order
        player.sord = player.participant.sorder
        player.ord_2 = player.participant.order_2
def set_payoff(player: Player):
    if player.subsession.round_number == player.payround:
        if int(player.true_y) == 1:
            prob = 1- (1 - player.assessment/100)*(1 - player.assessment/100)
        else:
            prob = 1 - (player.assessment/100)*(player.assessment/100)
        player.prob = prob
        if player.participant.treatment < 6:
            pay = float((4*np.random.choice([0,1], 1, p=[1-prob, prob])[0])) +1 #assessment based payment
        if player.participant.treatment == 3 or player.participant.treatment == 4 or player.participant.treatment == 5:
            if np.random.choice([0,1], 1, p=[0.5,0.5])[0] == 0:
                pay = float(4*player.participant.true_y) +1 #bonus based payment
        player.payoff = pay


# PAGES
class Instructions(Page):

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        player.prolific_id = player.participant.label

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.treatment < 6

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startexptime = int(time.time())
        return dict()

class Instructions_s(Page):
    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        player.prolific_id = player.participant.label

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startexptime = int(time.time())
        return dict()

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        player.prolific_id = player.participant.label

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.participant.treatment > 5


class Instructions_Sen(Page):
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.participant.treatment > 5

class Instructions_sen_pay_A(Page):
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.participant.treatment == 6

class Instructions_sen_pay_B(Page):
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.participant.treatment == 7

class Sender(Page):
    form_model = 'player'
    form_fields = ['t_selection', 'n_selection']

    @staticmethod
    def vars_for_template(player: Player):
        import time
        o_v = np.random.permutation([4, 5, 6, 7, 8, 9, 10, 11])
        o_h = np.random.permutation([0, 1, 2, 3])
        lr = np.random.permutation([0, 1])[0] #answer to question: Is det on the right?
        lr_hint = np.random.permutation([0, 1])[0] #answer to question: Is 2-state on top?
        o = np.append(o_h, o_v)  # reordering of rows
        o = np.append(o, np.array([12]))

        present = [int(player.sord[0]), int(player.sord[1]), int(player.sord[2])]     #order of these arrays: d_balanced, d_pro, d_con, s_balanced, s_pro, s_con
        # order of these arrays: d_balanced, d_pro, d_con, s_balanced, s_pro, s_con
        hb = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2])

        nb_d1 = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                         [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]]) #plus
        nb_d2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1]]) #minus
        nb_d3 = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]]) #join
        nb_s1 = np.array([[0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]]) #plus
        nb_s2 = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                         [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1]]) #minus
        nb_s3 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]) #join
        hb = hb[o] #hb is always the same
        d_plus = nb_d1[present[player.round_number-1]][o]
        d_minus = nb_d2[present[player.round_number-1]][o]
        d_join = nb_d3[present[player.round_number-1]][o]
        s_plus = nb_s1[present[player.round_number-1]][o]
        s_minus = nb_s2[present[player.round_number-1]][o]
        s_join = nb_s3[present[player.round_number-1]][o]
        x = player.round_number-1
        #randomize column order
        nb_d = [d_plus, d_minus, d_join]
        nb_s = [s_plus, s_minus, s_join]
        rand = np.random.permutation([0, 1, 2])
        if lr == 0: #then, deterministic table is table 1
            nb_1 = [nb_d[rand[0]], nb_s[rand[0]]]
            nb_2 = [nb_d[rand[1]], nb_s[rand[1]]]
            nb_3 = [nb_d[rand[2]], nb_s[rand[2]]]
        else:
            nb_1 = [nb_s[rand[0]], nb_d[rand[0]]]
            nb_2 = [nb_s[rand[1]], nb_d[rand[1]]]
            nb_3 = [nb_s[rand[2]], nb_d[rand[2]]]
        plus = np.where(rand==0)[0][0]+1
        join = np.where(rand==2)[0][0]+1
        player.starttime = int(time.time())

        #save player setup
        player.row_order = str(o)
        player.det_stoch_order = str(lr)
        player.state_order = str(lr_hint)
        player.column_order = str(np.array(['plus', 'minus', 'join'])[rand]) #0: plus, 1: minus, 2: join
        if present[x] == 0:
            player.table = 'balanced'
        if present[x] == 1:
            player.table = 'plus'
        if present[x] == 2:
            player.table = 'minus'

        if lr == 0: #in auswertung, replace 1-state with harmless hint and 2-state with hint, stoch and det are the same.
            if lr_hint == 0:
                u = ['In the historical data, it looks like there could be an effect of E'+str(plus)+' on the main observation (Y).', 'In the historical data, it looks like there could be an effect of E'+str(plus)+' on the main observation (Y).']
                d = ['If E'+str(plus)+' equals 1, the main observation (Y) also equals 1.', 'Almost always, if E'+str(plus)+' equals 1, the main observation (Y) also equals 1.']

            else:
                d = ['In the historical data, it looks like there could be an effect of E'+str(plus)+' on the main observation (Y).', 'In the historical data, it looks like there could be an effect of E'+str(plus)+' on the main observation (Y).']
                u = ['If E'+str(plus)+' equals 1, the main observation (Y) also equals 1.', 'Almost always, if E'+str(plus)+' equals 1, the main observation (Y) also equals 1.']
        else:
            if lr_hint == 0:
                u = ['In the historical data, it looks like there could be an effect of E'+str(plus)+' on the main observation (Y).', 'In the historical data, it looks like there could be an effect of E'+str(plus)+' on the main observation (Y).']
                d = ['Almost always, if E'+str(plus)+' equals 1, the main observation (Y) also equals 1.', 'If E'+str(plus)+' equals 1, the main observation (Y) also equals 1.']

            else:
                d = ['In the historical data, it looks like there could be an effect of E'+str(plus)+' on the main observation (Y).', 'In the historical data, it looks like there could be an effect of E'+str(plus)+' on the main observation (Y).']
                u = ['Almost always, if E'+str(plus)+' equals 1, the main observation (Y) also equals 1.', 'If E'+str(plus)+' equals 1, the main observation (Y) also equals 1.']

        return dict(
            hb=hb,
            nb_1=nb_1,
            nb_2=nb_2,
            nb_3=nb_3,
            round=player.round_number,
            plus = plus,
            join = join,
            lr = lr,
            table = player.table,
            lr_hint=lr_hint,
#            lr_1 = 1-lr,
            u = u,
            d = d
        )

    @staticmethod
    def is_displayed(player: Player):
        return player.participant.treatment > 5 and player.round_number < 4

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        import time
        player.finishtime = int(time.time())


class Quiz_Sen_A(Page):
    form_model = 'player'
    form_fields = ['quizs1', 'quizs2', 'quizs3', 'quizs4', 'quizs5', 'quizs6']

    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quizs1='False', quizs2='False', quizs3='True ', quizs4='True ', quizs5='...there is a 1.', quizs6='False')
        if values != solutions:
            return "One or more responses were unfortunately wrong."

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startquiztime = int(time.time())
        return dict()

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.participant.treatment == 6

class Quiz_Sen_B(Page):
    form_model = 'player'
    form_fields = ['quizs1', 'quizs2', 'quizs3', 'quizs4', 'quizs5', 'quizs6']

    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quizs1='False', quizs2='False', quizs3='True ', quizs4='True ', quizs5='...there is a 1.', quizs6='True ')
        if values != solutions:
            return "One or more responses were unfortunately wrong."

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startquiztime = int(time.time())
        return dict()

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.participant.treatment == 7


class Instructions_rec_NH(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and (player.treatment == 0 or player.treatment == 3)

class Instructions_rec_H(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and (player.treatment == 1 or player.treatment == 2 or player.treatment == 4 or player.treatment == 5)

class Instr_betw_NH(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 7 and (player.treatment == 0 or player.treatment == 3)

class Instr_betw_H(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 7 and (player.treatment == 1 or player.treatment == 2 or player.treatment == 4 or player.treatment == 5)

class Instructions_payoff_A(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.participant.treatment < 3

class Instructions_payoff_B(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.participant.treatment > 2 and player.participant.treatment < 6


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


        present = [int(player.ord[0]), int(player.ord[1]), int(player.ord[2]), int(player.ord[3]), int(player.ord[4]), int(player.ord[5])]     #order of these arrays: d_balanced, d_pro, d_con, s_balanced, s_pro, s_con
        present_2 = [int(player.ord_2[0]), int(player.ord_2[1]), int(player.ord_2[2]), int(player.ord_2[3]), int(player.ord_2[4]), int(player.ord_2[5])]


        # order of these arrays: d_balanced, d_pro, d_con, s_balanced, s_pro, s_con
        hb = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2])
        nb_1 = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                         [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]])  # plus
        nb_2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                         [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1]])  # minus
        nb_3 = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])  # join
        hb = hb[o]  # hb is always the same
        x = player.round_number - 1
        if (x < 6):
            y = present[x]
        else:
            y = present_2[x-6]  # this if stage 2 for deterministic!
        plus = nb_1[y][o]
        minus = nb_2[y][o]
        join = nb_3[y][o]

        if y == 0:
            player.table = 'det_balanced'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'If'
                keyword_2 = 'equals 1, the main observation also always equals 1.'
            else:
                keyword = 'In the historical data, it looks like there could be an effect of'
                keyword_2 = 'on the main observation.'
        if y == 1:
            player.table = 'det_plus'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'If'
                keyword_2 = 'equals 1, the main observation also always equals 1.'
            else:
                keyword = 'In the historical data, it looks like there could be an effect of'
                keyword_2 = 'on the main observation.'
        if y == 2:
            player.table = 'det_minus'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'If'
                keyword_2 = 'equals 1, the main observation also always equals 1.'
            else:
                keyword = 'In the historical data, it looks like there could be an effect of'
                keyword_2 = 'on the main observation.'
        if y == 3:
            player.table = 'stoch_balanced'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'Almost always, if'
                keyword_2 = 'equals 1, the main observation also always equals 1.'
            else:
                keyword = 'In the historical data, it looks like there could be an effect of'
                keyword_2 = 'on the main observation.'
        if y == 4:
            player.table = 'stoch_plus'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'Almost always, if'
                keyword_2 = 'equals 1, the main observation also always equals 1.'
            else:
                keyword = 'In the historical data, it looks like there could be an effect of'
                keyword_2 = 'on the main observation.'
        if y == 5:
            player.table = 'stoch_minus'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'Almost always, if'
                keyword_2 = 'equals 1, the main observation also always equals 1.'
            else:
                keyword = 'In the historical data, it looks like there could be an effect of'
                keyword_2 = 'on the main observation.'

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
            keyword=keyword,
            keyword_2=keyword_2
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
        return player.round_number < 7 and (player.treatment == 2 or player.treatment == 1 or player.treatment ==5 or player.treatment == 4)


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

        present = [int(player.ord[0]), int(player.ord[1]), int(player.ord[2]), int(player.ord[3]), int(player.ord[4]), int(player.ord[5])]
        present_2 = [int(player.ord_2[0]), int(player.ord_2[1]), int(player.ord_2[2]), int(player.ord_2[3]), int(player.ord_2[4]), int(player.ord_2[5])]

        # order of these arrays: d_balanced, d_pro, d_con, s_balanced, s_pro, s_con
        hb = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2])
        nb_1 = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                         [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]])  # plus
        nb_2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                         [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1]])  # minus
        nb_3 = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])  # join
        hb = hb[o]  # hb is always the same
        x = player.round_number - 1
        if (x < 6):
            y = present[x]
        else:
            y = present_2[x-6]  # this if stage 2 , same order
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
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'If'
                keyword_2 = 'and E' +str(joinloc) +' both equal 1 at the same time, the main observation (Y) also always equals 1.'
                exkeyword = 'If'
                exkeyword_2 = 'equals 1, the main observation (Y) also always equals 1.'
            else:
                keyword = 'If'
                keyword_2 = 'equals 1, the main observation (Y) also always equals 1.'
                exkeyword = 'In the historical data, it looks like there could be an effect of'
                exkeyword_2 = 'on the main observable.'
        if y == 1:
            player.table = 'det_plus'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'If'
                keyword_2 = 'and E' +str(joinloc) +' both equal 1 at the same time, the main observation (Y) also always equals 1.'
                exkeyword = 'If'
                exkeyword_2 = 'equals 1, the main observation (Y) also always equals 1.'
            else:
                keyword = 'If'
                keyword_2 = 'equals 1, the main observation (Y) also always equals 1.'
                exkeyword = 'In the historical data, it looks like there could be an effect of'
                exkeyword_2 = 'on the main observable.'
        if y == 2:
            player.table = 'det_minus'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'If'
                keyword_2 = 'and E' +str(joinloc) +' both equal 1 at the same time, the main observation (Y) also always equals 1.'
                exkeyword = 'If'
                exkeyword_2 = 'equals 1, the main observation (Y) also always equals 1.'
            else:
                keyword = 'If'
                keyword_2 = 'equals 1, the main observation (Y) also always equals 1.'
                exkeyword = 'In the historical data, it looks like there could be an effect of'
                exkeyword_2 = 'on the main observable.'
        if y == 3:
            player.table = 'stoch_balanced'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'Almost always, if'
                keyword_2 = 'and E' +str(joinloc) +' both equal 1 at the same time, the main observation (Y) also always equals 1.'
                exkeyword = 'Almost always, if'
                exkeyword_2 = 'equals 1, the main observation (Y) also always equals 1.'
            else:
                keyword = 'Almost always, if'
                keyword_2 = 'equals 1, the main observation (Y) also always equals 1.'
                exkeyword = 'In the historical data, it looks like there could be an effect of'
                exkeyword_2 = 'on the main observable.'
        if y == 4:
            player.table = 'stoch_plus'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'Almost always, if'
                keyword_2 = 'and E' +str(joinloc) +' both equal 1 at the same time, the main observation (Y) also always equals 1.'
                exkeyword = 'Almost always, if'
                exkeyword_2 = 'equals 1, the main observation (Y) also always equals 1.'
            else:
                keyword = 'Almost always, if'
                keyword_2 = 'equals 1, the main observation (Y) also always equals 1.'
                exkeyword = 'In the historical data, it looks like there could be an effect of'
                exkeyword_2 = 'on the main observable.'
        if y == 5:
            player.table = 'stoch_minus'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'Almost always, if'
                keyword_2 = 'and E' +str(joinloc) +' both equal 1 at the same time, the main observation (Y) also always equals 1.'
                exkeyword = 'Almost always, if'
                exkeyword_2 = 'equals 1, the main observation (Y) also always equals 1.'
            else:
                keyword = 'Almost always, if'
                keyword_2 = 'equals 1, the main observation (Y) also always equals 1.'
                exkeyword = 'In the historical data, it looks like there could be an effect of'
                exkeyword_2 = 'on the main observable.'


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
            row_order=player.row_order,
            keyword=keyword,
            keyword_2=keyword_2,
            exkeyword=exkeyword,
            exkeyword_2=exkeyword_2

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
        return player.round_number > 6 and (player.treatment == 2 or player.treatment == 1 or player.treatment ==5 or player.treatment == 4)

class Stage1_NH(Page):
    form_model = 'player'
    form_fields = ['assessment']

    @staticmethod
    def vars_for_template(player: Player):
        import time
        o_v = np.random.permutation([4, 5, 6, 7, 8, 9, 10, 11])
        o_h = np.random.permutation([0, 1, 2, 3])
        o = np.append(o_h, o_v)  # reordering of columns
        o = np.append(o, np.array([12]))

        present = [int(player.ord[0]), int(player.ord[1]), int(player.ord[2]), int(player.ord[3]), int(player.ord[4]), int(player.ord[5])]     #order of these arrays: d_balanced, d_pro, d_con, s_balanced, s_pro, s_con
        present_2 = [int(player.ord_2[0]), int(player.ord_2[1]), int(player.ord_2[2]), int(player.ord_2[3]), int(player.ord_2[4]), int(player.ord_2[5])]

        # order of these arrays: d_balanced, d_pro, d_con, s_balanced, s_pro, s_con
        hb = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2])
        nb_1 = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                         [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]])  # plus
        nb_2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                         [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1]])  # minus
        nb_3 = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])  # join
        hb = hb[o]  # hb is always the same
        x = player.round_number - 1
        if (x < 6):
            y = present[x]
        else:
            y = present_2[x-6]  # this if stage 2
        plus = nb_1[y][o]
        minus = nb_2[y][o]
        join = nb_3[y][o]

        if y == 0: #ugly, should use (everywhere) player.table = ['det_balanced', 'det_plus', 'det_minus', 'stoch_balanced', 'stoch_plus', 'stoch_minus'][y]
            player.table = 'det_balanced'
        if y == 1:
            player.table = 'det_plus'
        if y == 2:
            player.table = 'det_minus'
        if y == 3:
            player.table = 'stoch_balanced'
        if y == 4:
            player.table = 'stoch_plus'
        if y == 5:
            player.table = 'stoch_minus'

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
            row_order=player.row_order
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
        return player.round_number < 7 and (player.treatment == 0 or player.treatment ==3)


class Stage2_NH(Page):
    form_model = 'player'
    form_fields = ['assessment']

    @staticmethod
    def vars_for_template(player: Player):
        import time
        o_v = np.random.permutation([4, 5, 6, 7, 8, 9, 10, 11])
        o_h = np.random.permutation([0, 1, 2, 3])
        o = np.append(o_h, o_v)  # reordering of columns
        o = np.append(o, np.array([12]))

        present = [int(player.ord[0]), int(player.ord[1]), int(player.ord[2]), int(player.ord[3]), int(player.ord[4]), int(player.ord[5])]
        present_2 = [int(player.ord_2[0]), int(player.ord_2[1]), int(player.ord_2[2]), int(player.ord_2[3]), int(player.ord_2[4]), int(player.ord_2[5])]
        # order of these arrays: d_balanced, d_pro, d_con, s_balanced, s_pro, s_con
        hb = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2])
        nb_1 = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                         [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]])  # plus
        nb_2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                         [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1]])  # minus
        nb_3 = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])  # join
        hb = hb[o]  # hb is always the same
        x = player.round_number-1
        if (x < 6):
            y = present[x]
        else:
            y = present_2[x-6]  # this if stage 2 for deterministic!
        plus = nb_1[y][o]
        minus = nb_2[y][o]
        join = nb_3[y][o]

        if y == 0:
            player.table = 'det_balanced'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'If'
                keyword_2 = 'equals 1, the main observation also always equals 1.'
            else:
                keyword = 'In the historical data, it looks like there could be an effect of'
                keyword_2 = 'on the main observable.'
        if y == 1:
            player.table = 'det_plus'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'If'
                keyword_2 = 'equals 1, the main observation also always equals 1.'
            else:
                keyword = 'In the historical data, it looks like there could be an effect of'
                keyword_2 = 'on the main observable.'
        if y == 2:
            player.table = 'det_minus'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'If'
                keyword_2 = 'equals 1, the main observation also always equals 1.'
            else:
                keyword = 'In the historical data, it looks like there could be an effect of'
                keyword_2 = 'on the main observable.'
        if y == 3:
            player.table = 'stoch_balanced'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'Almost always, if'
                keyword_2 = 'equals 1, the main observation also always equals 1.'
            else:
                keyword = 'In the historical data, it looks like there could be an effect of'
                keyword_2 = 'on the main observable.'
        if y == 4:
            player.table = 'stoch_plus'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'Almost always, if'
                keyword_2 = 'equals 1, the main observation also always equals 1.'
            else:
                keyword = 'In the historical data, it looks like there could be an effect of'
                keyword_2 = 'on the main observable.'
        if y == 5:
            player.table = 'stoch_minus'
            if player.treatment == 2 or player.treatment == 5:
                keyword = 'Almost always, if'
                keyword_2 = 'equals 1, the main observation also always equals 1.'
            else:
                keyword = 'In the historical data, it looks like there could be an effect of'
                keyword_2 = 'on the main observable.'


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
        player.column_order= str(np.array(['plus', 'minus', 'join'])[rand]) #0: plus, 1: minus, 2: join

        return dict(
            hb=hb,
            nb_1=c_1,
            nb_2=c_2,
            nb_3=c_3,
            round=player.round_number,
            plus=plus,
            join=join,
            column_order = player.column_order,
            table = player.table,
            row_order=player.row_order,
            keyword=keyword,
            keyword_2 = keyword_2
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
        return player.round_number > 6 and (player.treatment == 0 or player.treatment ==3)


class Certainty(Page):
    form_model = 'player'
    form_fields = ['certainty']
    def is_displayed(player: Player):
        return player.treatment < 6
class Certainty_Sender(Page):
    form_model = 'player'
    form_fields = ['certainty_sen']
    def is_displayed(player: Player):
        return player.treatment > 5 and player.round_number == 3

class Quiz_A_NH(Page):
    form_model = 'player'
    form_fields = ['quiz1', 'quiz2', 'quiz4', 'quiz5']

    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quiz1='False', quiz2='False', quiz4='True', quiz5='False')
        if values != solutions:
            return "One or more responses were unfortunately wrong."

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.participant.treatment == 0

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startquiztime = int(time.time())
        return dict()

class Quiz_B_NH(Page):
    form_model = 'player'
    form_fields = ['quiz1', 'quiz2', 'quiz4', 'quiz5']

    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quiz1='False', quiz2='False', quiz4='True', quiz5='True')
        if values != solutions:
            return "One or more responses were unfortunately wrong."

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 1 and player.participant.treatment == 3

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startquiztime = int(time.time())
        return dict()

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
        return player.subsession.round_number == 1 and (player.participant.treatment == 2 or player.treatment == 1)

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
        return player.subsession.round_number == 1 and (player.participant.treatment == 5 or player.treatment == 4)

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startquiztime = int(time.time())
        return dict()

class Hint_Constr(Page):
    form_model = 'player'
    form_fields = ['hint_constr']
    def is_displayed(player: Player):
        return player.treatment < 6 and player.round_number == 6

class Quiz_betw_NH(Page):
    form_model = 'player'
    form_fields = ['quiz3', 'quiz6']

    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quiz3='True', quiz6='True')
        if values != solutions:
            return "One or more responses were unfortunately wrong."

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startquiz2time = int(time.time())
        return dict()

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 7 and (player.participant.treatment == 0 or player.participant.treatment == 3)

class Quiz_betw_H(Page):
    form_model = 'player'
    form_fields = ['quiz6']

    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(quiz6='True')
        if values != solutions:
            return "One or more responses were unfortunately wrong."

    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.startquiz2time = int(time.time())
        return dict()

    @staticmethod
    def is_displayed(player: Player):
        return player.subsession.round_number == 7 and (player.participant.treatment == 2 or player.participant.treatment == 1 or player.participant.treatment == 5 or player.participant.treatment == 4)

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


page_sequence = [Instructions, Instructions_s, Captcha, Disqual, Instructions_Sen, Instructions_rec_NH, Instructions_rec_H, Instructions_payoff_A, Instructions_payoff_B, Instructions_sen_pay_A, Instructions_sen_pay_B,Quiz_Sen_A, Quiz_Sen_B, Quiz_A_NH, Quiz_B_NH, Quiz_A_H, Quiz_B_H, Sender, Stage1_NH, Stage1_H, Instr_betw_NH, Instr_betw_H, Quiz_betw_NH, Quiz_betw_H, Stage2_NH, Stage2_H, Certainty, Certainty_Sender, Hint_Constr]
