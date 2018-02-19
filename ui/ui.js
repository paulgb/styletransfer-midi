'use strict';

const WS_URI = 'ws://paperspace:9000/';

class MIDIController {
    constructor(levelsCallback) {
        this.levelsCallback = levelsCallback;

        console.log('Connecting to controller.');
        navigator.requestMIDIAccess().then((midi) => {
            console.log('Got midi');
            var inputs = midi.inputs.values();
            console.log(inputs);
            for (var input = inputs.next(); input && !input.done; input = inputs.next()) {
                input.value.onmidimessage = this.midiMessageReceived.bind(this);
            }
        });
    }

    midiMessageReceived(e) {
        var index = e.data[1];
        var value = e.data[2];

        if (0 <= index && index < 8) {
            this.levelsCallback(index, value);
        } else if (16 <= index && index < 24) {
            this.levelsCallback(index - 8, value);
        }
    }
}

class StyleTrainer {
    constructor(websocketURI, callback) {
        this.ws = new WebSocket(websocketURI);
        this.ws.onmessage = this.onMessage.bind(this);
        this.callback = callback;
    }

    onMessage(e) {
        let data = JSON.parse(e.data);
        this.callback(data);
    }

    setWeight(weightIndex, value) {
        //console.log(weightIndex, value);
        this.ws.send(`${weightIndex} ${value}`);
    }
}

class MainController {
    constructor() {
        this.styleTrainer = new StyleTrainer(WS_URI, this.onStyleEpoch.bind(this));
        this.midiController = new MIDIController(this.onLevelsChange.bind(this));
    }

    onLevelsChange(index, value) {
        this.styleTrainer.setWeight(index, value);
    }

    onStyleEpoch(data) {
        console.log(data);

        var weightScale = d3.scaleLinear().domain([0, 127]).range(['0%', '100%']);
        var maxLoss = d3.max(data.layers, (d) => d3.max(d.parts, (t) => t.loss));
        var totalLoss = d3.sum(data.layers, (d) => d3.sum(d.parts, (t) => t.loss));
        var lossScale = d3.scaleLinear().domain([0, maxLoss]).range(['0%', '100%']);

        // Update loss.
        d3.select('#loss').text(`Loss: ${totalLoss.toPrecision(6)} Image: ${data.img}`);

        // Update image.
        d3.select('#result').attr('src', `${data.img}?${Math.random()}`);

        // Update metrics table.
        var partHeaders = d3.select('#part_names').selectAll('td').data(data.parts);
        partHeaders.enter().append('td').merge(partHeaders).text((d) => d);

        var res = d3.select('#metrics tbody').selectAll('tr').data(data.layers, (d) => d.name);
        res = res.enter().append('tr').call((d) => d.append('th').text((t) => t.name)).merge(res);

        var targ = res.selectAll('td').data((d) => d.parts);
        targ = targ.enter().append('td').classed('vis', true).call((t) =>
            t.append('div').append('div').classed('weight', true).text(' ')
        ).call((t) =>
            t.append('div').append('div').classed('loss', true).text(' ')
        ).merge(targ);

        targ.select('div.weight').datum((d) => d.weight).style('width', (d) => weightScale(d));
        targ.select('div.loss').datum((d) => d.loss).style('width', (d) => lossScale(d));
    }
}

new MainController();
